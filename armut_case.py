
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih




#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False) 
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_csv("Datasets/armut_data.csv")
df = df_.copy()
df.head()

def check_df(dataframe):
    print("############## Shape #############")
    print(dataframe.shape)
    print("############## Type #############")
    print(dataframe.dtypes)
    print("############## Head #############")
    print(dataframe.head())
    print("############## Tail #############")
    print(dataframe.tail())
    print("############## NA #############")
    print(dataframe.isnull().sum())
    print("############## Quantiles #############")
    print(dataframe.describe().T)

check_df(df)

df.describe().T

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = (df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str))

# Out[61]:
#         UserId  ServiceId  CategoryId           CreateDate Hizmet
# 0        25446          4           5  2017-08-06 16:11:00    4_5
# 1        22948         48           5  2017-08-06 16:12:00   48_5



# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df.dtypes

#CreateDate i time tipine dönüştürdüm.
for col in df.columns:
    if "Date" in col:
        df[col] = pd.to_datetime(df[col])

# Formatı year-month yaparak New_Date'e atadım.
df["New_Date"] = pd.to_datetime(df["CreateDate"],format='%Y-%m').dt.to_period('M')
df.head()




# SepetID = UserID_New_Date oluşturulacak

df["SepetID"] = (df["UserId"].astype(str) + "_" + df["New_Date"].astype(str))




#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

# not:pdf'te sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. der
# burada da aylık alınan hizmetler birden fazla. ör. kullanıcı aynı ay içinde 2 kez hizmet almış gibi. hizmetlere tam olarak erişebilmek için categoryID lerden gidicem.
# bunları pivot işleme dökersem; unstack() ile categoryID tanımlarını değişkenler olarak tanımlıyorum.
# yukarıdaki olması gereken çıktıyı görebilmek için iloc[0:5, 0:9] indeks bazlı seçim yapıyorum.

df.head()
df.groupby(["SepetID", "Hizmet"]).agg({"CategoryId": "count"}).unstack().iloc[0:5, 0:10]



# çıktıda değişkenler 1-0 olarak çıksın istiyorum. şöyle olsun: CategoryId den hizmet satın alırı saydır dolu olanlara 1, NaN değerlere 0 yaz.
df.groupby(["SepetID", "Hizmet"]).agg({"CategoryId": "count"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:10]



# bunu fonksiyonla ifade edelim.

def create_basket_service_df(dataframe, id=True): #id False olduğunda else koşulu çalışır.
    if id:
        return dataframe.groupby(["SepetID", "Hizmet"])['CategoryId'].count().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["SepetID", "Hizmet"])['ServiceId'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

bas_ser_df = create_basket_service_df(df) #[71220 rows x 50 columns]




# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(bas_ser_df, #apriori derki, yukarıda verileri işlediğimiz fonk. ismi üzerinden,dfteki aynı isimleri kullanmak istersen use_colnames=True yap. bende sana olası ürün birlikteliklerinin çıktılarını vereyim.
                            min_support=0.01, #teori kısmında min support değeri veriyorduk. bu o. support değeri %1in altında olmasın.
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False) #supporta göre azalan şekilde sırala.

#(18_4) kodlu hizmetin tek başına gözlenme olasılığı  0.238121

# Ama tam ihtiyacım olan bu değil. birliktelik kurallarını çıkarmam gerekiyor.
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
sorted_rules = rules.sort_values("lift", ascending=False)
#  antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
#10      (22_0)      (25_0)            0.047515            0.042895  0.011120    0.234043  5.456141  0.009082    1.249553
#11      (25_0)      (22_0)            0.042895            0.047515  0.011120    0.259247  5.456141  0.009082    1.285834
#19      (38_4)       (9_4)            0.066568            0.041393  0.010067    0.151234  3.653623  0.007312    1.129413

# 0. index için yorumlar:
# antecedents: önceki hizmet (22_0)
# consequents: ikinci hizmet (25_0)
# antecedent support: hizmetin tek başına gözlenme olasılığı 0.047515
# consequent support: 2.hizmetin tek başına gözlenme olasılığı 0.042895
# support:  antecedents: önceki hizmetin ve consequents: ikinci hizmetin birlikte gözükme olasılığı 0.011120
# confidence: X hizmeti alndığında Y nin alınması olasılığı  0.234043
# lift:  X hizmeti satın alndığında Y nin satın alınması olasılığı 5.456141 kat artar

# leverage: kaldıraç etkisi. lifte benzer. leverage değeri supportu yüksek olan değerlere öncelik verme egiliminde bundan dolayı ufak bir yanlılığı vardır.
# Lift değeri ise daha az sıklıkta olmasına ragmen bazı ilişkileri yakalayabilmektedir. bizim için daha değerlidir. yansızdır.
# conviction: Y hizmetin olmadan X hizmetinin beklenen değeri, frekansıdır. diğer taraftan X hizmeti olmadan Y hizmetinin beklenen frekansıdır.
#leverage ve conf.  değerine çok odaklanmıyoruz. lift conf. ile gidiyoruz.


df.info()
df.head()
df.shape

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz..tolist()


def arl_recommender(rules, ServiceId, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, Service in enumerate(sorted_rules["antecedents"]):
        for j in list(Service):
            if j == ServiceId:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules,"2_0",5)

# ['22_0', '25_0', '15_1', '13_11', '38_4']
# en son ServiceId:2 - CategoryId:0 olan bir hizmeti alan kişiye 5 tavsiyede bulunursak:
# 22,25,..,38 nolu ServiceId lerin sırasıyla 0,0,...,4 nolu CategoryId lerini alabilir deriz.


