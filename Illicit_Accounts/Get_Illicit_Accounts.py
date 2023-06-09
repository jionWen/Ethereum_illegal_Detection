
import datetime
import json
import requests
import numpy as np

"""
Scope: Get all documented scams using etherscamDB API
Method Info: Parse response as json file and retain relevant fields on which analysis shall be carried out
"""
def get_number_of_illicit_addresses():
    response = requests.get("https://etherscamdb.info/api/scams/")
    response = response.json()
    no_of_scams = len(response['result'])
    print(no_of_scams)

'''
利用web3库和etherscamDB提供的接口，获取非法账户，并得到对应JSON数据
'''
def get_illicit_account_addresses():
    from web3 import Web3
    response = requests.get("https://etherscamdb.info/api/scams/")  # GET请求
    if response.status_code == 200:
        response = response.json()
        no_of_scams = len(response['result'])
        scam_id, scam_name, scam_status, scam_category, addresses= ([] for i in range(5))
        #从得到的数据response中进行遍历获取非法账户的地址
        for scam in range(no_of_scams):
            if 'addresses' in response['result'][scam]:
                for i in response['result'][scam]['addresses']:
                    if i[:2] != '0x':
                        continue
                    addresses.append(i)
                    scam_id.append(response['result'][scam]['id'])    #非法地址ID
                    scam_name.append(response['result'][scam]['name'])  #非法地址Name
                    scam_status.append(response['result'][scam]['status']) #非法地址Status
                    if 'category' in response['result'][scam]:    #判断当前scam是否有类别标签
                        scam_category.append(response['result'][scam]['category'])
                    else:
                        scam_category.append('Null')
        print("非法账户的数量 ", len(addresses))
        print("对所得非法账户去重", len(np.unique(addresses)))
        return addresses

        # Total Number of Available documented scam addresses: 692 (19/02/2019)
        # JSON File
        address_darklist = json.loads(open('../illegal_lists/addresses-darklist.json').read())
        addresses_2 = []

        for item in address_darklist:
            addresses_2.append(item['address'])
        print("Number of illegal addresses: ", len(address_darklist))
        print("Number of unique illegal addresses in JSON file: ", len(np.unique(addresses_2)))

        all_addresses = np.concatenate((addresses, addresses_2), axis=None)  #将addresses和addresses_2拼接得到所有的非法地址
        all_addresses = np.unique(np.char.lower(all_addresses))
        print("Final number of unique Addresses: ", len(np.unique(all_addresses)))
        return all_addresses

"""
Scope: Additional documented scam/illicit behavior URLs
Method Info: Parse local json file and return as array
Total Number of Available documented scam URLs: 2370 (19/02/2019) 
Link: https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/urls/urls-darklist.json
"""
def get_additional_scam_websites():
    url_darklist = json.loads(open('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Illicit_Accounts/illegal_lists/urls-darklist.json', encoding="utf8").read())
    print("Number of illegal addresses: ", len(url_darklist))
    url, comments = ([] for i in range(2))
    for item in url_darklist:
        url.append(item['id'])
        comments.append(item['comment'])
    print(url[0], " ", comments[0])


"""
Scope: Get Current Ether Price and supply in terms of USD from etherscan
Method Info: Request price and supply info using Etherscan API
API Key: "1BDEBF8IZY2H7ENVHPX6II5ZHEBIJ8V33N"
Link: https://github.com/corpetty/py-etherscan-api/blob/master/examples/contracts/get_abi.py
"""
def get_Last_Ether_Price_Supply():
    from etherscan.stats import Stats
    with open("Illicit_Accounts/api_key.json", mode='r') as key_file:
        key = json.loads(key_file.read())['key']

    api = Stats(api_key=key)
    ether_last_price_json = api.get_ether_last_price()
    ether_btc = ether_last_price_json['ethbtc']
    ether_datetime = convertTimestampToDateTime(ether_last_price_json['ethbtc_timestamp'])
    ether_usd_price = ether_last_price_json['ethusd']
    #ether_usd_price_datetime = convertTimestampToDateTime(ether_last_price_json['ethusd_timestamp'])
    total_ether_supply = api.get_total_ether_supply()
    print("Time of price: ", ether_datetime, " Ether_BTC price: ", ether_btc, " Ether_USD price: ", ether_usd_price)
    print("Total Ether supply available: ", total_ether_supply)

def convertTimestampToDateTime(timestampValue):
    timestampValue = int(timestampValue)
    value = datetime.datetime.fromtimestamp(timestampValue)
    exct_time = value.strftime('%d %B %Y %H:%M:%S')
    return exct_time

def main():
    #get_number_of_illicit_addresses()
    get_illicit_account_addresses()
    #get_additional_scam_websites()
    #get_Last_Ether_Price_Supply()

if __name__ == '__main__':
    main()
