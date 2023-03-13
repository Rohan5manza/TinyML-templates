// The C++ library provides APIs to load the model, allocate memory for input and output tensors,creatw the input and output tensors,
//store data collected from sensors, into the input tensors
//and to run inference on the model
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/models.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <cstdio>

int main() 
{
constexpr char kModelFileName[]="model.tflite"; //storing name of file in constant char array
TfLiteModel* model=tflite::GetModel(kModelFileName);

if(model==nullptr)
{
    cerr<<"Error, model not found";
}

// Create an interpreter to run the model
tflite::AllOpsResolver resolver;
tfLite::MicroInterpreter interpreter(model,resolver);

//Allocate memory for the model's input and output tensors

interpreter.AllocateTensors(); //allocating memory for model's inputs and outputs

TfLiteTensor* input=interpreter.input(0);// pointer to model's input tensors, starting from 0th location

TfLiteTensor* output=interpreter.output(0);

// define input array for the model.Populate this array later with the sensor data collected 
const float input_data[4]={1.0,2.0,3.0,4.0};


// Run inference on input data 

float input_data[];

float read_sensor_data(input_data)
{
    // write a function to read data from sensors,and store it in the input tensor format
}


memcpy(input->data.raw,input_data,sizeof(input_data));
interpreter.Invoke();
float output_data=output->data.f[0];
}