# Workload file format
## 1. request.csv
- format: "city", "uri", "rating", "timestamp"
- explanation:
  - city: city id / client id
  - uri: uri id / content id
  - rating: rating value
  - timestamp: relative timestamp, in seconds.
- example:
```
43,181120,1,0
41,13631,1,0
34,30509,1,0
```

## 2. summarize.txt
- format: 5 lines, summary file
- explanation:
  - line 1: uri numbers, total content number
  - line 2: uri max, max content id
  - line 3: city numbers, total client number
  - line 4: total rows, request file line number, total request number
  - line 5: top 1000 uri popularity, with (uri id, popularity) pairs
- example:
```
uri numbers: 47677
uri max: 196611
city numbers: 50
total rows: 100000
Top 1000 uri popularity: [(141768, 0.8891968416376661), (66267, 0.8869584693595501), ...]
```

## 3. group_uri_dict.json
- format: json file, record a list of content IDs for each timestamp.
- explanation:
  - key: timestamp
  - value: uri list
- example:
```
{
  "3": [20, 13631, 30509,...],
  "22": [10, 18, 245,...],
  ...
}
```

## 4. ip_city_dict.json
- format: json file, map a client id to a city ID(control_domain_id).
- explanation:
  - key: client id
  - value: city id
- example:
```
{
  "0": 43,
  "1": 41,
  "2": 34,
  ...
}
```

## 5. df-1m-unique.csv
- format: "city", "uri", "rating", "timestamp"
- explanation: this file is used to train CDAE model.
  - city: city id / client id
  - uri: uri id / content id
  - rating: rating value
  - timestamp: relative timestamp, in seconds.
- example:
```
117257,13631,1,0
109636,181120,1,0
19913,30509,1,0
...
```

## 6. uri2code.csv
- format: "uri_name", "uri_code", this file is used to map uri name to uri code.
- example:
```
http://www.wengmq.top/,1
http://www.wengmq.top/0001bace1ae1daef2c3f2dd707cfd724/1655297914/2862/20210321/9f761f40f903e971f22047d9371f1c57,2
http://www.wengmq.top/0002075bd3dca5753a57514586e9acb7/1655248533/b72c/20201224/779f097e6bd25dd8063ded8607916fd4,3
...
```

## 7. city2code.csv
- format: "city_name", "city_code", this file is used to map city name to city code.
- example:
```
city_name,city_code
Baoqiao,1
Beijing,2
Changning,3
...
```
