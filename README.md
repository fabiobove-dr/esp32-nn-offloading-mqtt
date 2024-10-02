# ESP32 Neural Network Offloading with MQTT

This project demonstrates offloading neural network computations from an ESP32 device to an edge server using MQTT. The ESP32 device connects to a Wi-Fi network, registers itself on the edge server, and communicates through MQTT topics to perform neural network inference layer by layer.

## Features

- **Wi-Fi Connectivity:** The ESP32 connects to a specified Wi-Fi network.
- **MQTT Communication:** The ESP32 uses MQTT to send and receive messages from an edge server.
- **Neural Network Offloading:** Inference layers are run on the ESP32, and results are sent to the edge server.
- **TensorFlow Lite Micro:** The project utilizes TensorFlow Lite Micro for running machine learning models on the ESP32.
- **NTP Time Synchronization:** The device syncs with an NTP server for accurate timestamps.
- **UUID Generation:** Each message and device is uniquely identified using UUIDs.

## Dependencies

- **ESP32 Board Support Package** (via Arduino IDE)
- **Libraries:**
  - `WiFi.h`: For Wi-Fi connectivity.
  - `PubSubClient.h`: For MQTT communication.
  - `ArduinoJson.h`: For handling JSON data.
  - `tensorflow/lite/micro`: For running TensorFlow Lite models on microcontrollers.
  - `UUID.h`: For generating UUIDs.

## Configuration

Edit the `conf.h` file to set up the following configurations:

- **Wi-Fi Credentials:** `SSID` and `PWD`
- **MQTT Broker:** `MQTT_SRV`, `MQTT_PORT`
- **NTP Server:** `NTP_SRV`, `NTP_GMT_OFFSET`, `NTP_DAYLIGHT_OFFSET`

## Model Setup

The project supports up to 5 layers of a TensorFlow Lite model, which are loaded and run sequentially. Each layer is defined in its own header file (e.g., `layer_0.h`, `layer_1.h`).

Ensure that the TensorFlow Lite model layers are properly converted and placed in the `model_layers` directory.

## Topics

The following MQTT topics are used for communication:

- `devices/`: Device registration.
- `DeviceUUID/model_data`: Topic to send model data for inference.
- `DeviceUUID/model_inference`: Topic to trigger inference for a specific layer.
- `DeviceUUID/model_inference_result`: Topic to receive inference results.

## Running the Project

1. Flash the code to the ESP32.
2. Ensure that the MQTT broker is running and accessible.
3. The device will:
   - Connect to Wi-Fi.
   - Register itself on the edge server.
   - Subscribe to necessary MQTT topics.
   - Perform neural network inference layer by layer upon receiving input data and an offloading layer index.

## Example Workflow

1. **Device Registration:** The ESP32 registers itself to the edge server by publishing a registration message.
2. **Receive Input Data:** The device receives input data for the neural network through the `model_data` topic.
3. **Run Inference:** The device runs inference for each layer of the neural network and publishes the results to the `model_inference_result` topic.
4. **Publish Results:** Inference results, including the output and inference time for each layer, are published to the edge server.

## Notes

- The size of the MQTT packet is set to 3 KB (`MQTT_MAX_PACKET_SIZE`).
- The device restarts automatically after completing the test, ensuring fresh initialization for the next session.
- Ensure that the edge server is properly configured to handle incoming MQTT messages and process the neural network inference results.

## License

This project is licensed under
