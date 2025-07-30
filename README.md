# Saphire

Hybrid RFHE is a Julia library implementing a hybrid solution for faster FHE inference of CNNs.

## Parameters

By default parameters used are for 16 bits precision with a partial extraction level of 1 or 2. 
If you want to change the scheme parameters, they are defined at src/Constants.jl. The relevant parameters are discribed in the paper.
Please note that for M = 5^5 and M = 3^7 g=2 but that for M = 7^4, g=3.

## Usage

In the examples file you will find ready to use sample of code to launch an inference. Please use multi_run_resnet_cifar for optimal CPU performances. You can call it with the number of desired threads.

```julia --threads=nb_desired thread multi_run_resnet_cifar.jl ```

If you want to test for other partial extraction level and or model, you can just modify those variables at the begining of this file.

Note : The files run_imagenet and run_resnet are deprecated. 

A proof of concept GPU implementation is also made available here.