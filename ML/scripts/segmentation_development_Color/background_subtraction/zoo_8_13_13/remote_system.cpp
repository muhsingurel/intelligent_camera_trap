/*
 * Run the openCV background subtraction MOG2 on input frames
 * from a webcam.  Find the center of mass for images segmented
 * this way, move turret accordingly.
 * Save frames  from the webcam making a new folder every 5 minutes
 * in the directory specified in "file_directory"
 */

#include<opencv2/opencv.hpp>
#include<iostream>
#include <vector>
#include <time.h>
#include <string>
#include <cstdlib>
#include <math.h>
#include "motor_controller.h"

using namespace cv;
using namespace std;

#define PI 3.14159


/*
 * Return the centroid of a segmented binary image.
 */
Point getCentroid(cv::Mat img)
{
    Point coord;
    Moments mm = moments(img,false);
    double moment10 = mm.m10;
    double moment01 = mm.m01;
    double moment00 = mm.m00;
    coord.x = int(moment10 / moment00);
    coord.y = int(moment01 / moment00);
    return coord;
}


int main(int argc, char *argv[])
{
    Mat frame;
    Mat gray;
    Mat fore;

    const int FRAME_W = 160;
    const int FRAME_H = 120;

    string file_directory ("/media/SanDisk/Zoo_8_13_13/");
    string image_file_name;
    string make_directory;
    string current_directory;
    string current_frame;
    string first_file;
    const char* make_directory_c;
    
    BackgroundSubtractorMOG2 bg;
    Point last_center;
    unsigned long frame_count = 0;
    int x = 0;
    int y = 0;

    char current_time[24];
    char next_time_c[24];
    time_t rawtime;
    time_t start_time;
    time_t next_time;

    //namedWindow("Localization", CV_WINDOW_AUTOSIZE);

    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout << endl << "Failed to connect to the 360 camera." << endl << endl;
    }
    else
    {
        cout << endl << "Connected to 360 camera." << endl << endl;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_W);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_H);

    //make first image directory
    time(&start_time);
    strftime(current_time, 24, "%m-%d-%H:%M:%S",localtime(&start_time));
    make_directory = "mkdir -p " + file_directory + current_time;
    current_directory = file_directory + current_time + "/";
    make_directory_c = make_directory.c_str();
    system(make_directory_c);
    next_time = start_time + 300; //next directory should be 5 mins later
    
    MotorController mctrl("/dev/ttyUSB0", 19200, 0.0, 0.0);
    mctrl.updatePanTilt();
    cout << "Motor controller created" << endl;
    
    for(;;)
    {
        cap >> frame;
        //cout << frame_count; 
        time(&rawtime);
        strftime(current_time, 24, "%m-%d-%H:%M:%S",localtime(&rawtime));

        if( rawtime >= next_time )
        {
            strftime(next_time_c, 24, "%m-%d-%H:%M:%S",localtime(&next_time));
            make_directory = "mkdir " + file_directory + current_time;
            current_directory = file_directory + current_time + "/";
            make_directory_c = make_directory.c_str();
            system(make_directory_c);
            next_time += 300;  //next directory 5 mins after this one
            frame_count = 0;   //reset frame_count
        }
        
        cvtColor(frame, gray, CV_BGR2GRAY); //grayscale the frame
        bg.operator()(gray,fore);           //get the binary foreground image
        last_center = getCentroid(fore);    
        x += last_center.x;
        y += last_center.y;

        if (frame_count % 5 == 0)          //every .. frames, come up w/ a new tracking location"
        {
            x = int(x/5.);  //avg all our centroid coordinates
            y = int(y/5.);
            last_center.x = x;
            last_center.y = y;
            
            //MOTOR CONTROL HERE
            x = x - FRAME_W/2;
            y = -y + FRAME_H/2;

            //x = 0;
            //y = FRAME_H/2 - 10;
            if (y > FRAME_H/8)
            {
                double x_deg = (180 + (int)(atan2(y, x)*180/(PI))) % 180;
                //printf("offset = %f\n", x_deg);
                mctrl.new_pan = x_deg;
                mctrl.new_tilt = mctrl.tilt_pos;
                //cout << "Offset Calculated, sending command..." << endl;
                mctrl.updatePosition();
                //cout << "Sent" << endl;
            }
            else
            {
                //cout << "No significant offset" << endl;
            }
            x = 0;
            y = 0;
        }
        current_frame = to_string(frame_count);
        image_file_name = current_directory + current_time + "_" + current_frame + ".jpg";
        imwrite(image_file_name, frame);            

        ++frame_count;  

        if( waitKey(5) >= 0) break;
    }
return 0;
}
