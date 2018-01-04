# Style-Transfer

## CycleGAN
Paper -> [here](https://arxiv.org/pdf/1703.10593.pdf "here")  

Haircolor translation black into white or white into black. Of course, you can translate into arbitrary haircolor if you prepare.  
Input size:128×128

## StarGAN
Paper -> [here](https://arxiv.org/abs/1711.09020 "here")

Haircolor translation into multi colors(ex : white -> black, blue, pink....).   
But, this code is not completed.  
Concretely. ar first, haircolor translation is done correctly. But if the number of epochs is increased,  generator comes to output images which are similar to raw images (Therefore, haircolor doesn't vary). Making improvements are needed.
