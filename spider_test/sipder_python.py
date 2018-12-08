from lxml import html
from requests import get,post,Session
from bs4 import BeautifulSoup


#找到网络请求中的request header  请求的头文件
#头文件中保留请求来源Referer以及代理User-Agent即可
header={
    'Referer':'******Referer******',
    'User-Agent':'**********User-Agent********',
}

# requests的Session可以自动保持cookie，不需要自己维护cookie内容
s=Session()

# 先访问login页面，拿一部分cookies
R_1 = s.get("http://official.lixueweb.com/login",headers=header)
# 再携带cookie提交登录的post，因为网络请求中需要三个登录信息：_csrf_token，_username，_password，可以在From Data 中查看需要的登录信息
#_csrf_token一般设置的属性为hidden
#获取_csrf_token
tree = html.fromstring(R_1.text)
csrf_token = list(set(tree.xpath("//input[@name='_csrf_token']/@value")))[0]
#完善登录信息
payload={'_csrf_token':csrf_token,'_username':'_username','_password':'_password'}
#请求login_check
R_2 = s.post("http://official.lixueweb.com/login_check",headers=header,data=payload)
#获取目标页面
R_3 = s.get("http://official.lixueweb.com/",headers=header)
#获取目标页面上想获取的网页
r_4=s.get('http://official.lixueweb.com/exam-marking-image/answerpaper-list?isAll=1',headers=header)
r_4.encoding='utf-8'


#使用beautifulsoup 抓去目标图片

# bs = BeautifulSoup(r_4.content,'html.parser')
# all_href=bs.find_all('div',class_="box-body")
# for a in all_href:
#     m_url = a.find('a').get('href')
#     level=
#     print(a)
#     print(m_url)
# print(all_href)
# all_href=[l['src'] for l in all_href]
# print(all_href)



