# coding: utf-8


"""
sample.py

Written_by: Taichi Iki (taici.iki@gmail.com)
Created_at: 2018-11-07
"""

import os

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.variable as V
from chainer import serializers

from jaembed import JaEmbedID


class SampleModel(chainer.Chain):
    """
    サンプルのニューラルネットワークモデル
    """
    def __init__(
            self, 
            metadata_path=None, 
            embed_path=None, 
            special_tokens=None
        ):
        super(SampleModel, self).__init__()
        
        self.ignore_label = -1
        self.charvec_dim  = 128
        
        HeN = chainer.initializers.HeNormal
        with self.init_scope():
            self.embed = JaEmbedID(
                    embed_path, 
                    special_tokens, 
                    metadata_path,
                    initialW=HeN(fan_option='fan_out')
                )
    
    def __call__(self, xs):
        """
        埋め込みの取得サンプル
        xs: 文字列をid系列に変換した変数
        """
        
        emb_xs = self.embed(xs)

        # 固有の計算
        # ...
        # サンプルなので埋め込みをそのまま返す
        
        return emb_xs


def sample_train(model_dir, model_name):
    # 事前学習済みの文字ベクトル
    embed_path = 'data/sg_d128_w8_mc0_neg5_iter10_s0p001.pklb'

    # 追加学習するトークン
    SYMBOL_BOS = '<BOS>'
    SYMBOL_EOS = '<EOS>'
    SYMBOL_UNK = '<UNK>'
    sp_char_list = [SYMBOL_UNK, SYMBOL_BOS, SYMBOL_EOS]
    
    # 新しいモデルを作成
    # 学習済みベクトルデータのパスと学習させたいトークンを指定
    model = SampleModel(
            embed_path=embed_path,
            special_tokens=sp_char_list
        )
    # 辞書データの保存
    model.embed.store_metadata(os.path.join(model_dir, 'embed_meta.pklb'))
    
    # 固有の学習
    # ...

    # モデルの保存
    model.to_cpu()
    model_path = os.path.join(model_dir, model_name)
    serializers.save_npz(model_path, model)


def sample_prediction(model_dir, model_name):
    # モデルをロード
    # メタデータを指定
    metadata_path = os.path.join(model_dir, 'embed_meta.pklb')
    model = SampleModel(
            metadata_path=metadata_path,
        )
    model_path = os.path.join(model_dir, model_name)
    serializers.load_npz(model_path, model)

    ID_EOS = model.embed.token2id['<EOS>']
    ID_UNK = model.embed.token2id['<UNK>']
    
    # 「こんにちは。」を埋め込みベクトルに変換する
    text = 'こんにちは。'

    x = model.xp.asarray([
            [model.embed.token2id.get(c, ID_UNK) for c in text] + [ID_EOS]
        ])
    print('x', x)
    print('embed', model(x))


if __name__ == '__main__':
    model_dir  = 'sample'
    model_name = 'sample.model'
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    sample_train(model_dir, model_name)
    sample_prediction(model_dir, model_name)
