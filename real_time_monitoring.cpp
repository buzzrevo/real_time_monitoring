#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <pylon/PylonIncludes.h>
#include <torch/script.h>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <nlohmann/json.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>



// initialize global variables
// create an alias for the websocketpp clien namespace
typedef websocketpp::client<websocketpp::config::asio_client> client;
client ws_client;

// create a mutex to check for thread safety
//std::map<std::string, double> latest_print_stats;
nhlohmann::json latest_print_stats;
std::mutex latest_print_stats_mutex;

void on_message(websocketpp::connection_hdl hdl, client::message_ptr msg) {
    nhlohmann::json data = nlohmann::json::parse(msg->get_payload());
    if (data.contains("result")) {
        //latest_print_stats['filament_used'] = data["result"]["jobs"][0][''filament_used'];
        std::lock_guard<std::mutex> lock(latest_print_stats_mutex);
        latest_print_stats = data["result"]["jobs"][0];
    } 
    // else {
    //     latest_print_stats["filament_used"] = std::nan("")
    // }
    ws.client.send(hdl,nlohmann::json({{"jsonrpc", "2.0"}, {"method", "server.history.list"}, {"params", {}}, {"id", 5656}}).dump(),websocketpp::frame::opcode::text);
}

void run_websocket_client() {
    // Intialize ASIO transport policy
    ws_client.init_asio();

    // Set the message handler
    ws_client.set_message_handler(&on_message);

    // Set the open handler to print a message
    ws_client.set_open_handler([](websocketpp::connection_hdl hdl) {
        std::cout << "Connected to the server" << std::endl;
    });

    // Also do so for closing
    ws_client.set_close_handler([](websocketpp::connection_hdl hdl) {
        std::cout << "Disconnected from the server" << std::endl;
    });

    // Create a connection to the server
    std::string uri = "ws://192.168.1.20:7125/websocket";
    websocketpp::lib::error_code ec;
    client::connection_ptr con = ws_client.get_connection(uri, ec);
    if (ec) {
        std::cout << "Could not create connection because: " << ec.message() << std::endl;
        return;
    }

    // Connect to the server
    ws_client.connect(con);

    // Start ASIO event loop; this will block until the connection is closed
    ws_client.run();
}

struct ReadoutInfo {
    std::string mp4_filename;
    std::vector<double> filament_used_by_image;
    std::vector<double> filament_used_by_image_timing;
    std::vector<int> prediction_by_image;
    double duration;
    double frame_rate;

    // Serialization function
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & mp4_filename;
        ar & filament_used_by_image;
        ar & filament_used_by_image_timing;
        ar & prediction_by_image;
        ar & duration;
        ar & frame_rate;
    }
};

void capture_and_analyze_video(const std::string& avi_filename, const std::string& mp4_filename, const std::string& readout_filename, int duration, int frame_rate) {
    Pylon::PylonInitialize();
    Pylon::CInstantCamera camera(Pylon::CTLFactory::GetInstance().CreateFirstDevice());
    camera.Open();
    camera.PixelFormat.SetValue("RGB8");
    camera.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);

    // Define video codec and create VideoWriter object
    int frame_width = camera.Width.GetValue();
    int frame_height = camera.Height.GetValue();
    cv::VideoWriter out(avi_filename, cv::VideoWriter::fourcc('M','J','P','G'), frame_rate, cv::Size(frame_width, frame_height));

    // Load model
    auto module = torch::jit::load("vgg16_printer_monitor_gcode_indicator_03_8.pth");
    module.eval();

    // Initialize containers
    std::vector<int> prediction_by_image;
    std::vector<double> filament_used_by_image;

    // Main loop
    while (camera.IsGrabbing()) {
        auto grab_result = camera.RetrieveResult(5000, Pylon::TimeoutHandling_ThrowException);
        if (grab_result.GrabSucceeded()) {
            // Changes affect buffer
            cv::Mat frame(frame_height, frame_width, CV_8UC3, (uint8_t*)grab_result.GetBuffer());
            cv::flip(frame, frame, 0);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            // Convert to PIL image
            cv::Mat inpute_image;
            cv::resize(frame, input_image, cv::Size(224, 224));
            input_image.convertTo(input_image, CV_32FC3, 1.0f / 255.0f);

            // Model  eval
            auto input_tensor = torch::from_blob(input_image.data, {1, 224, 224, 3});
            auto output = module.forward({input_tensor}).toTensor();

            // Get the predicted class
            auto max_result = output.max(1);
            prediction_by_image.push_back(std::get<1>(max_result).item<int>());

            // Update filatement used
            std::lock_guard<std::mutex> lock(latest_print_stats_mutex);
            auto filament_used = latest_print_stats["filament_used"].get<double>();
            filament_used_by_image.push_back(filament_used);

            cv::imshow("Original Frame", frame);
            out.write(frame);

            if (cv::waitKey(1) == 'q') break;
        }
        grab_result.Release();
    }
    // Release the camera and video writer
    camera.StopGrabbing();
    camera.Close();
    out.release();
    cv::destroyAllWindows();

    // Convert AVI to MP4 with subprocess
    std::string ffmpeg_command = "ffmpeg -i " + avi_filename + " -c:v libx264 -preset ultrafast -pix_fmt yuv420p " + mp4_filename;
    system(ffmeg_command.c_str());

    
    // Create the readout_info struct and populate it with data
    ReadoutInfo readout_info = {
        mp4_filename,
        filament_used_by_image,
        prediction_by_image,
        duration,
        frame_rate
    };

    // Serialize the struct to a binary file
    std::ofstream ofs("readout_filename.bin", std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << readout_info;
}

std::string getCurrentTimeStr() {
    // Get the current time
    std::time_t now = std::time(nullptr);
    std::tm* local_time = std::localtime(&now);

    // Create a string stream to format the time
    std::ostringstream oss;
    oss << std::put_time(local_time, "%Y%m%d-%H%M%S");

    // Return the formatted time string
    return oss.str();
}

int main() {
    std::thread ws_thread(run_websocket_client);

    // Get the current time string
    std::string timestr = getCurrentTimeStr();
    std::string output_directory = "C:\\Users\\revon\\OneDrive - Revo Foods\\tmp\\data\\raw";
    std::string avi_filename = output_directory + "\\output" + timestr + ".avi";
    std::string mp4_filename = output_directory + "\\output" + timestr + ".mp4";
    std::string readout_filename = output_directory + "\\output" + timestr;
    int duration = 160;
    int frame_rate = 10;

    // Capture video and analyze
    capture_and_analyze_video("output.avi", "output.mp4", "readout.txt", 10, 30);  

    // Join the websocket thread
    ws_thread.join();

    return 0;
}
