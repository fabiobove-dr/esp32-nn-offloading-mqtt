/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* DEPENDENCIES LIBS
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include <WiFi.h>
#include <PubSubClient.h>
#define MQTT_MAX_PACKET_SIZE 3*1024 // Important: Adjust size to correctly send and recieve topic messages
#include <sys/time.h>
#include <UUID.h>
#include <ArduinoJson.h>

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* LIBS for TFLITE
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#define MODEL_NAME "test_model"
#include "model_layers/layer_0.h"
#include "model_layers/layer_1.h"
#include "model_layers/layer_2.h"
#include "model_layers/layer_3.h"
#include "model_layers/layer_4.h"

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*  CONFIGURATIONS & GLOBAL VARIABLES
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include "conf.h"

// Communication & Offloading Variables
WiFiClient                  espClient;
PubSubClient                client(espClient);
bool                        deviceRegistered = false;
UUID                        uuid;
String                      MessageUUID = "";
String                      DeviceUUID = "";
StaticJsonDocument<2048>    jsonDoc;

String                      end_computation_topic;
String                      device_registration_topic = "devices/";
String                      model_data_topic;
String                      model_inference_topic;
String                      model_inference_result_topic;

bool                        testFinished = false;
bool                        modelDataLoaded = false;
int                         imageHeight = 10;
int                         imageWidth = 10;
float                       modelInferenceData[10][10] = {};

// Neural Network Variables
const int                   MAX_NUM_LAYER = 5;
tflite::MicroErrorReporter  micro_error_reporter;
tflite::ErrorReporter*      error_reporter = &micro_error_reporter;
const tflite::Model*        model = nullptr;
tflite::MicroInterpreter*   interpreter = nullptr;
TfLiteTensor*               input;
TfLiteTensor*               output;
constexpr int               kTensorArenaSize = 12*1024;
uint8_t                     tensor_arena[kTensorArenaSize];
bool                        modelLoaded = false;
bool                        firstInferenceDone = false; 

/*
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* TIMER CONFIGURATION & FLOATING-POINT TIMESTAMP
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
String getCurrTimeStr(){
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time_t currentTime = tv.tv_sec;
  int milliseconds = tv.tv_usec / 1000;
  int microseconds = tv.tv_usec % 1000000;
  char currentTimeStr[30];
  snprintf(currentTimeStr, sizeof(currentTimeStr), "%ld.%03d%03d", currentTime, milliseconds, microseconds);
  String currentTimeString = String(currentTimeStr);
  return currentTimeString;
}

void timeConfiguration(){
  // Configure NTP time synchronization
  configTime(NTP_GMT_OFFSET, NTP_DAYLIGHT_OFFSET, NTP_SRV);
  Serial.println("Connecting to NTP Server");
  // Try obtaining the time until successful
  struct tm timeinfo;
  while (!getLocalTime(&timeinfo)) {
    delay(500);
  }

  // Print current time
  Serial.println("NTP Time Configured - Current Time: ");
  Serial.println(getCurrTimeStr());
  return;
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* PUBLISH DEVICE PREDICTION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void publishDevicePrediction(int offloading_layer_index, JsonArray layer_output, JsonArray layers_inference_time) {    // Generate the JSON message
    jsonDoc["timestamp"] = getCurrTimeStr();
    jsonDoc["message_id"] = MessageUUID;
    jsonDoc["device_id"] = DeviceUUID;
    jsonDoc["message_content"] = JsonObject();
    jsonDoc["message_content"]["layer_output"] = layer_output;
    jsonDoc["message_content"]["offloading_layer_index"] = offloading_layer_index;
    jsonDoc["message_content"]["layers_inference_time"] = layers_inference_time;
    // Serialize the JSON document to a string
    String jsonMessage;
    serializeJson(jsonDoc, jsonMessage);
    // Publish the JSON message to the topic
    client.publish(model_inference_result_topic.c_str(), jsonMessage.c_str(), 2);
    Serial.println("Published Prediction");
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* LOAD NN LAYER
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void loadNeuralNetworkLayer(String layer_name){
  // Import del modello da testare -> Nome nell'header file
  if(layer_name.equals("layer_0"))model = tflite::GetModel(layer_0);
  if(layer_name.equals("layer_1"))model = tflite::GetModel(layer_1);
  if(layer_name.equals("layer_2"))model = tflite::GetModel(layer_2);
  if(layer_name.equals("layer_3"))model = tflite::GetModel(layer_3);
  if(layer_name.equals("layer_4"))model = tflite::GetModel(layer_4);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model provided is schema version not equal to supported!");
      return;
  } else { Serial.println("Model Layer Loaded!"); }

  // Questo richiama tutte le implementazioni delle operazioni di cui abbiamo bisogno
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  Serial.print("Interprete ok");

  // Alloco la memoria del tensor_arena per i tensori del modello
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
      Serial.println("AllocateTensors() failed");
      return;
  } else { Serial.println("AllocateTensors() done"); }
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* INFERENCE FOR NN LAYER
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
extern "C" void runNeuralNetworkLayer(int offloading_layer_index, float inputData[10][10]) {
  // Assuming inputData is in the format expected by your neural network
  for (int i = 0; i <= offloading_layer_index; i++) {
    String layer_name = "layer_" + String(i);
    float inizio = micros();
    
    loadNeuralNetworkLayer(layer_name); // Load the appropriate layer
    input = interpreter->input(0);

    // Copy the input data to the input tensor
    memcpy(input->data.f, inputData, imageHeight * imageWidth * sizeof(float)); // Adjust depending on your input format

    // Run inference
    interpreter->Invoke();

    // Store inference time in seconds
    jsonDoc["layer_inference_time"][i] = (micros() - inizio) / 1000000.0; // Convert microseconds to seconds

    // Extract relevant information from the output tensor
    TfLiteTensor* outputTensor = interpreter->output(0);
    float* outputData = outputTensor->data.f;
    // TODO: Fix this We need to take thee last element of data not always 1
    int numOutput = outputTensor->dims->data[1];

    /*
    Serial.println("Output tensor dimensions for layer: " + String(i));
    for (int d = 0; d < outputTensor->dims->size; d++) {
        Serial.print("Dimension ");
        Serial.print(d);
        Serial.print(": ");
        Serial.println(outputTensor->dims->data[d]);
    }
    */

    // Store output values in the JSON document
    for (int j = 0; j < numOutput; j++) {
      jsonDoc["layer_output"][i].add(String(outputData[j]));
    }

    Serial.println("Computed layer: " + String(i) + " Inf Time: " + String((micros() - inizio) / 1000000.0) + " s");
  }
  Serial.println("Last layer output: "+jsonDoc["layer_output"][offloading_layer_index].as<String>());
  publishDevicePrediction(offloading_layer_index, jsonDoc["layer_output"][offloading_layer_index], jsonDoc["layer_inference_time"]);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* WIFI CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void wifiConfiguration(){
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PWD);
  Serial.println("Connecting to WiFi...");
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println(".");
    delay(500);
    ESP.restart();
  } 
  Serial.println("Connected to WiFi - IP Address: ");
  Serial.println(WiFi.localIP());
  delay(500);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* MQTT CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void mqttConfiguration(){
  client.setServer(MQTT_SRV, MQTT_PORT);
  while (!client.connect("ESP32Client", "", "")) {
    Serial.println("Connecting to MQTT Broker");
    if (!client.connected()) {
      Serial.println("Failed to connect to MQTT Broker - retrying, rc=");
      Serial.println(client.state());
      delay(500);
    }
  }
  client.setBufferSize(2*1024); // Important: Adjust size to correctly send and recieve topic messages
  Serial.println("Connected to MQTT Broker");
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * GENERATE MESSAGE UUID
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void generateMessageUUID(){
  // Generate a UUID
  unsigned long seed = esp_random();
  uuid.seed(seed);
  uuid.setRandomMode();
  uuid.generate();
  MessageUUID = (String)uuid.toCharArray();
  MessageUUID = MessageUUID.substring(0, 4);
  DeviceUUID = "device_01"; // + MessageUUID;
 }

/*
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* REGISTER THE DEVICE ON THE EDGE
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void registerDevice(){
  // Generate the JSON message
  jsonDoc["timestamp"] = getCurrTimeStr();
  jsonDoc["message_id"] = MessageUUID;
  jsonDoc["device_id"] = DeviceUUID;
  jsonDoc["message_content"] = "HelloWorld!";
  // Serialize the JSON document to a string
  String jsonMessage;
  serializeJson(jsonDoc, jsonMessage);
  // Publish the JSON message to the topic
  client.publish(device_registration_topic.c_str(), jsonMessage.c_str(), 2);
  Serial.println("Device Registered");
  deviceRegistered = true;
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* GET MODEL DATA FOR PREDICTION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include <ArduinoJson.h>  // Make sure to include the ArduinoJson library

void getModelDataForPrediction(const String& inputData) {
  // Convert the inputData string to a 2D array
  try {
    for (int i = 0; i < imageHeight; ++i) {
      for (int j = 0; j < imageWidth; ++j) {
        modelInferenceData[i][j] = inputData[i * imageWidth + j] - '0'; // Assuming inputData contains numeric characters
      }
    }
    Serial.println("Model input data received");
  } catch (const std::exception& e) {
    Serial.print("Error receiving model input data: ");
    Serial.println(e.what());
  }
  modelDataLoaded = true;
}

void processIncomingMessage(char* topic, byte* payload, unsigned int length) {
  // Convert the incoming message to a string
  String message;
  message.reserve(length); // Reserve space in advance for efficiency
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  // Parse the JSON message and store it in the DynamicJsonDocument
  DynamicJsonDocument doc(3 * 1024); // Adjust size as needed
  DeserializationError error = deserializeJson(doc, message);
  
  // Check for parsing errors
  if (error) {
    Serial.print("JSON parsing error: ");
    Serial.println(error.c_str());
    return;
  }
  Serial.print("Recieved message:");
  Serial.println(topic);

  // Check if the message is for model_data
  if (strcmp(topic, model_data_topic.c_str()) == 0) {
    String inputData = doc["input_data"]; 
    getModelDataForPrediction(inputData);
  }
  
  // Check if the message is for model_inference
  if (strcmp(topic, model_inference_topic.c_str()) == 0) {
    int offloading_layer_index = doc["offloading_layer_index"];
    String inputData = doc["input_data"]; 
    getModelDataForPrediction(inputData);
    runNeuralNetworkLayer(offloading_layer_index, modelInferenceData);
  }

  // Check if the test is finished
  if (strcmp(topic, end_computation_topic.c_str()) == 0) {
    Serial.print("Ending Computation");
    testFinished = true;
  }

}

void dispatchCallbackMessages() {
  // Set the topics
  end_computation_topic = DeviceUUID + "/end_computation";
  model_data_topic = DeviceUUID + "/model_data";
  model_inference_topic = DeviceUUID + "/model_inference";
  model_inference_result_topic = DeviceUUID + "/model_inference_result";

  // Subscribe to the topic
  client.subscribe(model_data_topic.c_str());
  client.subscribe(model_inference_topic.c_str());
  client.subscribe(end_computation_topic.c_str());

  // Set the callback function
  client.setCallback([](char* topic, byte* payload, unsigned int length) {
    processIncomingMessage(topic, payload, length);
  });
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * SETUP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void setup() {
  Serial.begin(115200);
  wifiConfiguration();          // Wi-Fi Connection
  mqttConfiguration();          // MQTT
  timeConfiguration();          // Synchronize Timer - NTP server
  generateMessageUUID();        // Generate an Identifier for the message
  dispatchCallbackMessages();   // Set the callback function for the MQTT messages
  registerDevice();             // Register the device on the edge
}

/* 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * LOOP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void loop() {
  client.loop(); 
  if(testFinished){
    delay(10000);
    ESP.restart();
  }
}