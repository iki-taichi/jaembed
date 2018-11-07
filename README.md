# jaembed
JaEmbedID - A Japanese Character Embedding for Chainer

Chainer用の学習済み日本語文字埋め込み

学習済み(固定)トークンと学習可能トークンをハイブリットで使用可能


## 説明

JaEmbedIDは、[例文集合を入力とするニューラルネットワークを用いた文のスタイル変換に向けて](https://jsai-slud.github.io/sig-slud/)の文字埋め込み手法を再現した日本語文字埋め込みの実装と学習済みモデルです。

文字埋め込みのためモデル全体の容量は小さくしつつ、ベクトルに変換可能な文字種は広く取ることができます。

ASCIIコード(0~127)、JIS第1水準、JIS 第2水準、JIS非漢字の合計7666文字を対象として、[日本語版wikipedia](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%A4%E3%83%B3%E3%83%9A%E3%83%BC%E3%82%B8)の全記事ダンプを用いて学習し、出現しなかった文字を除外した7341文字に関する文字埋め込みベクトルを提供します。

文字埋め込みベクトルの学習には[gensimライブラリのWord2Vec](https://radimrehurek.com/gensim/models/word2vec.html)を用い、ベクト
ル次元128、窓幅8、10エポックの条件のもと学習しています。


## 環境

chainerに依存します。

Version 4.2.0 での動作を確認しています。


## 使い方

dataディレクトリとjaembed.pyを使用環境にコピーし、Chainerのlinkとして他のchainから呼び出します。

sample.pyでは、文字列の埋め込みをベクトルを取得するサンプルを示します。

```:python
import chainer
from jaembed import JaEmbedID

class SampleModel(chainer.Chain):
    def __init__(
            self, 
            metadata_path=None, 
            embed_path=None, 
            special_tokens=None
        ):
        super(SampleModel, self).__init__()
        
        self.charvec_dim  = 128
        
        HeN = chainer.initializers.HeNormal
        with self.init_scope():
            self.embed = JaEmbedID(
                    embed_path, 
                    special_tokens, 
                    metadata_path,
                    initialW=HeN(fan_option='fan_out')
                )
```

### 初期化時


```:python
# 学習済みベクトルデータのパスと学習させたいトークンを指定
model = SampleModel(
        embed_path='data/sg_d128_w8_mc0_neg5_iter10_s0p001.pklb',
        special_tokens=['<BOS>', '<EOS>', '<UNK>']
    )

# 辞書データの保存
model.embed.store_metadata('sample_dir/embed_meta.pklb')

# モデル保存
model.to_cpu()
serializers.save_npz('sample_dir/sample.model', model)
```

### モデルロード時

```:python
# メタデータを指定
model = SampleModel(
        metadata_path='sample_dir/embed_meta.pklb',
    )
serializers.load_npz('sample_dir/sample.model', model)
```

なお、学習済みベクトルはchain自体に保存されるため、モデルロード時にdataディレクトリは必要ありません。


## 学習済みモデルのライセンス

学習に使用した日本語版wikipediaの記事は基本的にCC-BY-SA 3.0のもとで公開されていますが、[Wikipedia:ウィキペディアを二次利用する](https://ja.wikipedia.org/wiki/Wikipedia:%E3%82%A6%E3%82%A3%E3%82%AD%E3%83%9A%E3%83%87%E3%82%A3%E3%82%A2%E3%82%92%E4%BA%8C%E6%AC%A1%E5%88%A9%E7%94%A8%E3%81%99%E3%82%8B)の二次利用方法の概要の記載

> ウィキペディア上の素材を適法に二次利用するには、主に以下の3種類の方法に従うことになります。
> 1. ライセンスに従って二次利用する。
>
>     〜略〜
>
> 2. 著作権の制限規定に従って二次利用する。
> 
>    ウィキペディア上の素材は、適用される著作権法が定める著作権制限規定に従って二次利用できます。この方法によれば、ライセンスの利用許諾条項に従う必要はありません。
> 
>     〜略〜
>
> 3. 権利者の許諾を受けて二次利用する。
>
>     〜略〜
>

の2ならびに、著作権法第47条の7(情報解析のための複製等）に従い、学習済みモデル含めMITライセンスで公開できると考えています。
