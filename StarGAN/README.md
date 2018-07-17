# StarGAN

## Summary
![stargan](https://github.com/SerialLain3170/GAN-papers/blob/master/makegirlsmoe/stargan.png)

CycleGANはシングルドメイン間でしか変換出来なかった。多ドメイン間では、また別のネットワークを用意する必要があった。  
そこで多ドメイン間でも変換できるようにしたのがこのStarGAN。
- Generatorは、どのドメインへと変化させるか、そのドメインを表すベクトルをconcatして代入する
- Discriminatorは、real/fakeの判別だけでなく、どのドメインかを分類することも行う(ACGANと同様?)

## Usage
データとデータラベルを`.npy`ファイルを予め用意しておいて以下のコマンドを実行。
```bash
$ python stargan.py --nc_size <DOMAIN>
```
- `DOMAIN`に今回変換するドメインの数を指定する。デフォルトは6にする、理由は下記。

## Experiment
- 論文中ではDiscriminatorとGeneratorの全層に対してInstance Normalizationを用いているが、学習が進まなかったためBatch Normalizationを用いている、
- また、ドメイン数であるが私の環境では7以上は以下のようにうまく変換出来なかった、論文中では勿論7以上でも出来ているのでそういう意味では不完全である。
![domain7](https://github.com/SerialLain3170/Style-Transfer/blob/master/StarGAN/result_3.png)

## Result
私の環境で生成した画像を以下に示す。今回は髪色変換を行った。
![image](https://github.com/SerialLain3170/Style-Transfer/blob/master/StarGAN/result_2.png)
- Input size : 128 * 128
- バッチサイズは24
- ドメイン数は6
- 最適化手法はAdam(α=0.0001, β1=0.5)
- ドメインを表すベクトルは、対応するindexの要素だけ1にしている(例えばドメイン数3の場合は
