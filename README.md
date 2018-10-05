# Style-Transfer
スタイル変換をGANを用いて行う。
## Dataset
[safebooru](https://safebooru.org/)から髪色でタグ検索を行い、各ドメイン2000枚ずつ集めた。  
前処理としては[こちら](https://github.com/SerialLain3170/Illustration-Generator)で述べているのと同じ

## Quick Results
### CycleGAN
Paper -> [here](https://arxiv.org/pdf/1703.10593.pdf "here")  
Input size:128×128  
![CycleGAN](./CycleGAN/result.jpg)

### StarGAN
Paper -> [here](https://arxiv.org/abs/1711.09020 "here")  
Input size:128×128  
![StarGAN](./StarGAN/result_2.png)

### AdaIN
Paper -> [here](https://arxiv.org/pdf/1703.06868.pdf)

![AdaIN](https://github.com/SerialLain3170/Style-Transfer/blob/master/AdaIN/images/anime.png)
