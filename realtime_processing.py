import numpy as np
import time
import onnxruntime
from scipy.io import wavfile
import os
import soundfile as sf

block_len=512
block_shift=128


rnnmodel=onnxruntime.InferenceSession('./model/rnn2.onnx')
model_input_name= [inp.name for inp in rnnmodel.get_inputs()]
model_input = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in rnnmodel.get_inputs()}
directory  = "E:/speech enhancement/dataset/testset/"
file_paths=[]
for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
testlist=file_paths
outpath='E:/speech enhancement/realtimetest/'
for path in testlist:
    S=path.split('/') 
    wavname=S[-1]
    rate, audio = wavfile.read(path)
    audio=audio/np.max(np.abs(audio))
    out_file = np.zeros((len(audio)))
# create buffer
    in_buffer = np.zeros((block_len)).astype('float32')
    out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks  
    start_time = time.time()   
    for idx in range(num_blocks):
         in_buffer[:-block_shift] = in_buffer[block_shift:]
         in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
         in_block=np.reshape(in_buffer,(1,1,-1)).astype('float32')
         model_input[model_input_name[0]] = in_block
         model_output = rnnmodel.run(None, model_input)
         out_block=model_output[0]
         model_input[model_input_name[1]] = model_output[1]  
         out_buffer[:-block_shift] = out_buffer[block_shift:]
         out_buffer[-block_shift:] = np.zeros((block_shift))
         out_buffer  += np.squeeze(out_block)
    # write block to output file
         out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    
    sum=time.time()-start_time
    
    print(sum)
    out_file=out_file.astype('float32')
    wavfile.write(outpath+wavname, 16000, out_file)
    print(len(audio)/16000)