import time
from pprint import pprint

from telethon.sync import TelegramClient
from telethon import functions
import csv

api_id = 29878946
api_hash = 'accdc8738b5491f6d19a5dd2c5d1f206'

categories_to_channels = {
    # "Экономика": ["@omyinvestments", "@financelist", "@centralbank_russia", "@banksta", "@financiru"],
    # "Технологии": ["@abulaphia", "@it_teech", "@d_code", "@prohitec", "@habr_com"],
    # "Политика": ["@RVvoenkor", "@SolovievLive", "@maximkatz", "@dmitrynikotin", "@medvedev_telegram"],
    # "Шоубиз": ["@SVETSKlE_HRONIKI", "@yobajur", "@spletni_tg", "@neopra_blin"],
    # "Крипта": ["@cryptogram_ton", "@crypto_mozgi", "@buff_10", "@crypto_aIex", "@NFT_Reality"],
    # "Путешествия/релокация": ["@aviasales", "@uletet2021", "@OneTwoTrip", "@tutu_travel", "@Guideofrelocation"],
    # "Образовательный контент": ["@toplesofficial", "@nauka_dnya", "@faktiru", "@nsmag", "@nplusone"],
    # "Развлечения": ["@dvachannel", "@mudak", "@internetpasta", "@Reddit"],
    # "Новости и СМИ": ['@meduzalive', '@rian_ru', '@breakingmash', '@moscowmap', '@novosti_voinaa'],
    # "Психология": ['@labkovskiy', '@psycholog_alexandr_shahov', '@satyadas_official', '@olga_korobeynikova_quantum', '@GilevMD'],
    # "Искусство": ['@oteatre', '@artgallery', '@pic_history', '@arkhlikbez', '@dancelab'],
    # "Спорт": ["@sportosnews", "@sportazarto", "@Match_TV", "@sportsru", "@KOnOfff"],
    # "Цитаты": ["@evodays", "@GoTodayGo", "@worldthought", "@world_thoughts", "@knigikultura"],
    # "Еда и кулинария": ["@recipe_zuma", "@eda2021", "@topretsept", "@dauotkysit", "@po_receptj"],
    # "Общее": []
    "Цитаты": ["@worldthought", ""],

}
# pprint(list(categories_to_channels.keys()))
with TelegramClient('AINews', api_id, api_hash) as client:
    news = []
    for category, channels in categories_to_channels.items():
        print(category)
        for channel in channels:
            print(channel)
            limit, curr_id = 100, 0
            res = []
            try:
                for _ in range(10):
                    result = client(functions.messages.GetHistoryRequest(
                        peer=channel, offset_id=curr_id, offset_date=0,
                        add_offset=0, limit=limit, max_id=0, min_id=0, hash=0,
                    ))
                    result = result.messages
                    print(len(result))
                    if len(result) == 0:
                        break
                    for i in range(len(result)):
                        if result[i].message and result[i].message != '':
                            nic = result[i].message.replace("\n", "").replace(";", ",")
                            res.append([nic, category, channel])
                    curr_id = result[-1].id
            except BaseException as e:
                print(e)
            time.sleep(1)
            news.extend(res)
    with open('dataset.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        for elem in news:
            writer.writerow(elem)

# news = [["?? Полицейские раскрыли преступную группу, члены которой обманом похитили у Газпромбанка и ВТБ более 150 миллионов рублей.Члены преступного сообщества создали несколько номинальных организаций, куда фиктивно устраивали работников. Потом эти «сотрудники» обращались в банки, чтобы заключить кредитный договор.Среди членов группы было две сотрудницы банков: одна работала в Газпромбанке, другая — в ВТБ. Они и помогали вносить заведомо ложные данные для оценки платёжеспособности клиента. В итоге фиктивные работники получали кредиты и не выплачивали долги.Следователи считают, что с начала 2019 года преступники смогли заработать 159 миллионов рублей. Группа преимущественно состояла из женщин. Двух из них — 40-летнюю Юлию Светлову и 37-летнюю Алёну Жданову — следствие считает организаторами преступного сообщества. @financelist", "Политика"]]
# with open('test999.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file, delimiter=';')
#     for elem in news:
#         writer.writerow(elem)
# token = "6461688817:AAEE79o3gUECz0vs1CRIMd8Sf1eWxuX27_g"
#
# for category, channels in categories_to_channels.items():
#     for channel_id in channels:
#         channel_info = f"https://api.telegram.org/bot{token}/getChat?chat_id=@{channel_id}"
#         chat_info_response = requests.get(channel_info)
#         chat_info_json = chat_info_response.json()
#         chat_id = chat_info_json['result']['id']
#         print(chat_info_response, chat_id)
#         url = f"https://api.telegram.org/bot{token}/getHistory?chat_id=@{chat_id}&limit=1000"
#         response = requests.get(url)
#         print(response)
