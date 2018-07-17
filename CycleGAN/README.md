# CycleGAN

## Summary
![cyclegan](https://github.com/SerialLain3170/GAN-papers/blob/master/makegirlsmoe/cyclegan.png)

従来のスタイル変換では、UNet+Adversaril Lossのようにパラレルデータが必要であった、そこでノンパラレルデータでもできるようにしたのがこのCycleGAN  
- 二組のGeneratorとDiscriminatorを用意
- Adversarial lossに加え、再変換された値とのl1 lossを見るcycle-consistency lossを考慮。

## Usage
変換前と変換後に対応するデータを`.npy`ファイルとして用意する。その後、
```bash
$ python cyclegan.py --lam2 <WEIGHT>
```
で実行。WEIGHTでAdversarial lossの重みを指定する。  

## Result
私の環境で実行した結果を示す。今回はキャラクターの髪色変換として用いた。  
![image](https://github.com/SerialLain3170/Style-Transfer/blob/master/CycleGAN/result.jpg)
- Input size : 128 * 128
- バッチサイズ10
- 最適化手法はAdam(α=0.002, β1=0.5)
- Adversarial lossの重みは3.0
