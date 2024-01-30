
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage
import time
import re
import json
import csv
def gen_datasets(words,places):
    llm=ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    datasets=[]
    for i in range(0,2):
        for word in words:
            system_content="你是一个数据撰写专家,会根据要求生成一条用于训练语言模型成为虚拟女友的中文数据"
            change="要求体现女友'{}'的特点".format(word)
            human_content=change+"""
            请给我一个数据示例，数据内容的得体，会让心灵体会到温暖,对话使用第一、二人称。
            要求返回json格式,
            格式要求为

            {
                man:
                girlfriend:
            }
            """

            message=[
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ]
            print(message)
            try:
                dataset=llm(message).content
                dataset=json.loads(dataset)

            except Exception as e:
                print(f"请求失败: {e}")
                time.sleep(20)
                continue

            print(dataset)
            q = dataset["man"]
            a = dataset["girlfriend"]
            example={}
            example["q"]=q
            example["a"]=a

            datasets.append(example)
            time.sleep(20)

    print(datasets)
    csv_file_path='./static/datasets2.csv'
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(csvfile)

        # 写入表头
        csv_writer.writerow(["man", "wemen"])

        # 写入数据
        for data in datasets:
            csv_writer.writerow([data["q"], data["a"]])

def test():
    csv_file_path = 'datasets.csv'

    # 读取 CSV 文件
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        # 创建 CSV reader 对象
        csv_reader = csv.reader(csvfile)

        # 读取表头
        header = next(csv_reader)
        print("表头:", header)

        # 读取数据
        for row in csv_reader:
            print("数据:", row)
if __name__=="__main__":
    import os
    os.environ["OPENAI_API_KEY"] = '#########################################################'
    girlfriend_features = [
        "温柔", "体贴", "聪明", "善解人意", "幽默", "关心", "细心", "善良", "独立", "包容",
        "乐观", "理解", "浪漫", "真诚", "积极", "懂得照顾人", "喜欢旅行", "对艺术感兴趣", "对科技有了解",
        "热爱阅读", "爱好健身", "擅长烹饪", "有品位", "梦想追求者", "喜欢冒险", "对动物友好", "热爱大自然",
        "充满活力", "对音乐敏感", "具有创造力", "喜欢学习新事物", "具备沟通能力", "注重健康", "有责任心",
        "具备决断力", "对家庭重视", "喜欢笑", "追求平衡", "有一颗感恩的心", "有人情味", "时尚", "追求品质生活",
        "有修养", "有主见", "尊重他人", "待人真诚", "懂得娱乐", "具备社交技能", "喜欢品味美食"
    ]
    conversation_scenarios = [
        "表达情感支持", "日常问候", "分享生活琐事", "询问彼此的一天", "分享有趣的笑话", "提供鼓励和正能量",
        "讨论最近的新闻", "谈论电影和电视节目", "分享音乐推荐", "聊聊天气", "谈论美食和烹饪", "推荐好书",
        "交流关于旅行的经验", "分享关于艺术的见解", "聊一些科技趋势", "分享关于健身的建议", "提供热门游戏推荐",
        "讨论有趣的文化现象", "分享激励人心的名言", "聊聊星座和占星术", "提供一些小贴士和生活技巧",
        "回顾一天中的亮点", "讨论未来的计划和目标", "分享心情日记", "推荐适合学习的资源", "聊一些科学知识",
        "分享关于动物的趣闻", "讨论最喜欢的体育", "分享对大自然的热爱", "提供一些有趣的游戏和玩法",
        "讨论彼此的兴趣爱好", "分享一些旅游景点", "谈论关于时尚的看法", "提供护肤和美容建议", "推荐影视剧情",
        "分享关于音乐的心情", "讨论彼此的工作和职业", "提供放松和冥想的建议", "聊聊关于家庭的话题",
        "分享对未来的期望", "提供有趣的冷知识", "讨论彼此的梦想", "分享一些感人的故事", "提供休闲娱乐建议",
        "聊一些关于社交的心得", "推荐有趣的社交活动", "分享对彼此的思念之情",
        "发送一个温馨的拥抱", "鼓励彼此对自己好一点", "分享感恩之心", "提醒彼此善待自己", "问候彼此有个美好的一天",
        "表达对彼此的感谢", "分享一首温柔的歌曲", "提供一个甜蜜的微笑", "谈论一些治愈的事物", "鼓励彼此宠爱自己",
        "传递一些甜蜜的秘密", "讨论有趣的约会点子", "分享彼此的小幸福", "暗示一些浪漫的时刻", "提供一些调皮的表白",
        "讨论最喜欢的爱情电影", "分享彼此的心动瞬间", "提供一些甜蜜的夜间问候", "鼓励一些亲密的对话", "分享关于彼此的梦幻"
    ]

    gen_datasets(girlfriend_features,conversation_scenarios)
    test()



