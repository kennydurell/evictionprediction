import json, csv

import requests

def json_zips(data):
    response = requests.post('https://us-street.api.smartystreets.com/street-address?auth-id=AUTH_ID&auth-token=AUTH_TOKEN', headers = {'Content-Type': 'application/json','Host': 'us-street.api.smartystreets.com'}, json=data)
    final_data = response.json()
    street_dict={}
    for address in final_data:
        street_dict[address[u'delivery_line_1']] = address[u'components'][u'zipcode']
    # print street_dict
    with open('bar.csv', 'ab') as csvfile:
        fieldnames = ['address', 'zip_code']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for key,value in street_dict.iteritems():
            writer.writerow({'address': key, 'zip_code': value})

    return street_dict
    
if __name__ == '__main__':
    data = [
  {
    "street": "87 Central Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "87 Niagara Avneue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "876 Huron Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "877 Ingerson Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "880 Oak Road",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "89 Belgrave Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "89 Niagra",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "890 Geary Boulevard",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "891 Post Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "896 Chesnut Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "9 1/2 Imperial Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "9 Castle Manor",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "9 Graces Drive",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "90 Park Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "90 Parkridge",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "90 Santacruz Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "900 Athens",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "900 Oak Road",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "901 Capitol",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "901 Holyoke Court",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "907 Sovanness Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "91 Belcher Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "912 Jacksons Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "914 Hamilton Road",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "915 Pierce Place",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "916 Rockdal Drive",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "926 Grove",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "927 Greenwich",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "929 Broderick Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "93 Stanyan Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "930  Bay",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "934 Broadway Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "936 Mission Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "940 Lawton Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "942 Potrero",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "945 Corbett Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "945 Hayes Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "948 Moscow",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "948 Rhodeisland Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "948 Visitation Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "95 Ledyard",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "950 Capitol Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "950 Quintara Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "954 Key Avenue",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "957 Hayes",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "958 Plymouth Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "969 Hayesst St",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "970 Key Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "971 North Point",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "973 1/2 2 Church",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "974 Broadway Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "979 Sutter",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "98 Chanery Lane",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "980 Green Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "980 Holloway Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "988 Pines Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "998 Hollister Street",
    "city": "San Francisco",
    "state": "CA"
  },
  {
    "street": "1400 Jones Street",
    "city": "San Francisco",
    "state": "CA"
  }
]
    final_data = json_zips(data)
