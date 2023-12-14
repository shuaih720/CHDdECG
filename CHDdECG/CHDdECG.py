#CHDdECG
from layers_Resblock import InputConv
from layers_Resblock import ResBlock
import tensorflow as tf
from layers_Transformer import EncoderLayer
from layers_TemporalAttention import TemporalAttention
from layers_Tabnet import TabNet
from layers_Downsampling import TabNet_downsampling
#define 形式定义model
# def ECGnet(num_classes,input_signal,input_clinical,input_wavelet):
def ECGnet(num_classes):
    #signal
    input_signal = tf.keras.Input(shape=(5000,9),name='input_signal')
        
    Signal_block_conv = InputConv(filter_num=32,kernel_size=30,stride=2,name='signal_block_conv')(input_signal)
    
#     Signal_block_res1 = ResBlock(filter_num=32,kernel_size=15,stride=2,name='signal_block_res1')(Signal_block_conv)
        
    Signal_block_res2_1 = ResBlock(filter_num=16,kernel_size=3,stride=2,name='signal_block_res2_1')(Signal_block_conv)
    Signal_block_res3_1 = ResBlock(filter_num=16,kernel_size=3,stride=2,name='signal_block_res3_1')(Signal_block_res2_1)
    Signal_block_res4_1 = ResBlock(filter_num=16,kernel_size=3,stride=2,name='signal_block_res4_1')(Signal_block_res3_1)
        
    Signal_block_res2_2 = ResBlock(filter_num=16,kernel_size=5,stride=2,name='signal_block_res2_2')(Signal_block_conv)
    Signal_block_res3_2 = ResBlock(filter_num=16,kernel_size=5,stride=2,name='signal_block_res3_2')(Signal_block_res2_2)
    Signal_block_res4_2 = ResBlock(filter_num=16,kernel_size=5,stride=2,name='signal_block_res4_2')(Signal_block_res3_2)
        
    Signal_block_res2_3 = ResBlock(filter_num=16,kernel_size=7,stride=2,name='signal_block_res2_3')(Signal_block_conv)
    Signal_block_res3_3 = ResBlock(filter_num=16,kernel_size=7,stride=2,name='signal_block_res3_3')(Signal_block_res2_3)
    Signal_block_res4_3 = ResBlock(filter_num=16,kernel_size=7,stride=2,name='signal_block_res4_3')(Signal_block_res3_3)
    
    model_con = tf.keras.layers.Concatenate(axis = -1,name = 'conv_concat')([Signal_block_res4_1,Signal_block_res4_2,Signal_block_res4_3])
    
    Signal_block_trans = EncoderLayer(d_model=48,num_heads=8, dff=128, dropout_rate=0.2,name='signal_block_trans')(model_con)
    
    Signal_temporal_attention = TemporalAttention(feature_dim=48,kernel_size=3,stride=1,name = 'signal_tem_atten')(Signal_block_trans)
    
    fla = tf.keras.layers.Flatten(name='signal_fla')(Signal_temporal_attention)
    
    Signal_block_tab1 = TabNet_downsampling(num_features=960,feature_dim=256,output_dim=128,num_decision_steps=0,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='signal_block_tab')(fla)[0]#num_features=960
    
    input_clinical = tf.keras.Input(shape=114,name='input_clinical')
    
    Clinical_block_tab1 = TabNet(num_features=114,feature_dim=256,output_dim=64,num_decision_steps=2,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='clinical_block_tab')(input_clinical)[0]
    
    input_wavelet = tf.keras.Input(shape=54,name='input_wavelet')
    
    Wavelet_block_tab1 = TabNet(num_features=54,feature_dim=256,output_dim=32,num_decision_steps=2,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='wavelet_block_tab')(input_wavelet)[0]
    
    fusion_block = tf.keras.layers.Concatenate(axis = -1,name = 'fusion_concat')([Signal_block_tab1,Clinical_block_tab1,Wavelet_block_tab1])
    
    tab1_all = TabNet(num_features=224,feature_dim=512,output_dim=256,num_decision_steps=2,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='tab_con')(fusion_block)[0]
    
    fusion_block_1 = tf.keras.layers.Dense(64 , activation='relu',name='fusion_dense1')(tab1_all)
    fusion_block_2 = tf.keras.layers.Dropout(rate = 0.2,name='fusion_dropout1')(fusion_block_1)
    fusion_block_3 = tf.keras.layers.Dense(num_classes, activation='softmax',name='fusion_dense2')(fusion_block_2)
    
    model = tf.keras.models.Model([input_signal,input_clinical,input_wavelet],fusion_block_3)
    
    return model
