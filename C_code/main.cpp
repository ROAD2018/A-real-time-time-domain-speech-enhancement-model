
#include <stdio.h>

#include "def.h"

#include <cstdlib>



#include "AudioFile.h"

#include <chrono>

void trnnOnlineimpl::__ExportWAV(
		const std::string & Filename, 
		const std::vector<float> & Data, 
		unsigned SampleRate){
	
	AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(16000);
	File.save(Filename, AudioFileFormat::Wave);			
						
}


void denoise() {

    //tflite model;
    TfLiteModel *filternet;
    //model options
    TfLiteInterpreterOptions *netoptions;
    //interpreter model
    TfLiteInterpreter *interpreterfilter;
    //input data of model
    TfLiteTensor *input_data_tensor;
    //input state of model
    TfLiteTensor *input_state_tensor;
    const TfLiteTensor *output_data_tensor;
    const TfLiteTensor *output_state_tensor;

    std::string wavNames[9] = {"/wav/0_1.wav",
                        "/wav/0_2.wav",
                        "/wav/0_3.wav",
                        "/wav/1_1.wav",
                        "/wav/1_2.wav",
                        "/wav/1_3.wav",
                        "/wav/3_1.wav",
                        "/wav/3_2.wav",
                        "/wav/3_3.wav"};
    std::string testoutNames[9] = {"/exp/0_1.wav",
                        "/exp/0_2.wav",
                        "/exp/0_3.wav",
                        "/exp/1_1.wav",
                        "/exp/1_2.wav",
                        "/exp/1_3.wav",
                        "/exp/3_1.wav",
                        "/exp/3_2.wav",
                        "/exp/3_3.wav"};


    clock_t start,endt;  //use to calculate procssing time of model
    std::vector<float>  testenhanceddata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputfile;
    filternet=TfLiteModelCreateFromFile(MODULE_NAME);  //load tflite model
    netoptions = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(netoptions, 2);
    interpreterfilter=TfLiteInterpreterCreate(filternet, netoptions);
    if(interpreterfilter==nullptr){
        std::cout<<"failed to create interpreter"<<std::endl;
        return -1;

    }
    if(TfLiteInterpreterAllocateTensors(interpreterfilter)!=kTfLiteOk){
        std::cout<<"failed to allocate tensor"<<std::endl
        return -1;

    }

    input_data_tensor=TfLiteInterpreterGetInputTensor(interpreterfilter, 0);
    input_state_tensor=TfLiteInterpreterGetInputTensor(interpreterfilter,1);
    output_data_tensor=TfLiteInterpreterGetOutputTensor(interpreterfilter, 0);
    output_state_tensor=TfLiteInterpreterGetOutputTensor(interpreterfilter, 1);
    

    for(int i = 0; i < 9; i++){
        inputfile.load(wavNames[i]);
        inputfile.printSummary();
        start=clock();
        int k;
        int audionum=inputfile.getNumSamplesPerChannel();
        int blocknum=audionum/BLOCK_SHIFT;
        float audiodata[BLOCK_LEN]={0};
        float inputstate[statesize]={0};
        float outputaudio[BLOCK_LEN]={0};
        for(int j=0;j<blocknum-1;j++){
            memmove(audiodata,audiodata+BLOCK_SHIFT,(BLOCK_LEN-BLOCK_SHIFT)*sizeof(float));
            for(int n=0;n<BLOCK_SHIFT;n++){
                audiodata[n+BLOCK_LEN-BLOCK_SHIFT]=inputfile.samples[0][n+j*BLOCK_SHIFT];
            } 
            TfLiteTensorCopyFromBuffer(input_data_tensor,audiodata,BLOCK_LEN*sizeof(float));
            TfLiteTensorCopyFromBuffer(input_state_tensor,inputstate,statesize*sizeof(float));
            if(TfLiteInterpreterInvoke(interpreterfilter)!=kTfLiteOk){
                    std::cout<<"error invoke mode"<<std::endl;
            }
            float testflattenout[BLOCK_LEN]; 
            TfLiteTensorCopyToBuffer(output_data_tensor, testflattenout,BLOCK_LEN*sizeof(float)); //copy output data to array
            TfLiteTensorCopyToBuffer(output_state_tensor,inputstate,statesize*sizeof(float));
            memmove(outputaudio,outputaudio+BLOCK_SHIFT,(BLOCK_LEN-BLOCK_SHIFT)*sizeof(float));
            memset(outputaudio+BLOCK_LEN-BLOCK_SHIFT,0,BLOCK_SHIFT*sizeof(float));
            for (int t=0;t<BLOCK_LEN;t++ ){
                outputaudio[t] +=testflattenout[t];
            }
            for(int k=0;k<BLOCK_SHIFT;k++){
                    testenhanceddata.push_back(outputaudio[k]);
                }

            }
        endt=clock();
        double duration=(endt-start)/CLOCKS_PER_SEC;
        double audiotime=audionum/16000;
        std::cout<<duration/audiotime<<std::endl;
        __ExportWAV(testoutNames[i],testenhanceddata,16000);
        std::cout<<testoutNames[i]<<std::endl;
        testenhanceddata.clear();
    }

}





