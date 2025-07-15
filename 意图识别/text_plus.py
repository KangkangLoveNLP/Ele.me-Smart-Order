import numpy as np
import pandas as pd
from nlpcda import Similarword
smw = Similarword(create_num=1, change_rate=0.3)
def tongyici(sentence):
    rs1 = smw.replace(sentence)
    return rs1

from nlpcda import Homophone
smw_yin = Homophone(create_num=1, change_rate=0.3)
def tongyinci(sentence):
    rs1 = smw_yin.replace(sentence)
    return rs1

from nlpcda import RandomDeleteChar
smw_shan = RandomDeleteChar(create_num=1, change_rate=0.3)
def shanchu(sentence):
    rs1 = smw_shan.replace(sentence)
    return rs1


from nlpcda import baidu_translate

# 申请你的 appid、secretKey
# 两遍洗数据法（回来的中文一般和原来不一样，要是一样，就不要了，靠运气？）
def transfor(sentence):
    en_s = baidu_translate(content=sentence, appid='20250429002346026', secretKey='cVp3errYgul3Yhsi61pw',t_from='zh', t_to='en')
    zh_s = baidu_translate(content=en_s, appid='20250429002346026', secretKey='cVp3errYgul3Yhsi61pw',t_from='en', t_to='zh')
    return zh_s

def yuyin(sentence):
    return sentence

path_all = 'E:/laoren/train_with_zao.txt' 

from tqdm import tqdm
import time
if __name__ == "__main__":
    data = pd.read_csv('train.txt',encoding='utf-8',sep='\t')
    data_n = data.sample(frac=0.15,random_state=42)
    df = data_n.reset_index(drop=True)
    text_zao = []
    dom = []
    types = []
    for index,i in tqdm(df.loc[:,["text",'dom']].iterrows()):
        c = 1
        one = len(df)//5
        if index < one:
            re = tongyici(i['text'])
            type_o = 1
        elif index > one and index < one*2:
            re = tongyinci(i['text'])
            type_o = 2
        elif index > one*2 and index < one*3:
            re = shanchu(i['text'])
            type_o = 3
        elif index > one*3 and index < one*4: 
            x = 0
            while c == 1:
                print(f"第{index}个，重试{x}次")
                try: 
                    re = transfor(i['text'])
                    type_o = 4
                    c = 0
                except:
                    x+=1
                    time.sleep(2)
        else:
            re = yuyin(i['text'])
            type_o = 5
        text_zao.append(str(re))
        dom.append(i['dom'])
        types.append(type_o)
    for index,i in tqdm(data.loc[:,["text",'dom']].iterrows()):
        text_zao.append(i["text"])
        dom.append(i['dom'])
        types.append(0)        
    redf = pd.DataFrame(
        {'text':text_zao,'dom':dom,'type':types}
    )
    redf.to_csv(path_all, index=False, sep='\t')