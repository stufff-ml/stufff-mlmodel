require 'net/http'
require 'uri'
require 'json'
require 'date'
require 'csv'

# ruby retailrocket.rb http://stufff-review.appspot.com/api/1/events xoxo-...

#timestamp,visitorid,event,itemid,buyid
#1433224396283,536254,view,267839,
#1433222276276,599528,buy,356475,4000
#1433223239808,1377281,view,251467,
#1433223236124,287857,add,5206,

endpoint = ARGV[0]
token = ARGV[1]

filename = 'retailrocket.csv'
batch_size = 500
use_timestamp = false

# prepare the connection
uri = URI.parse endpoint

http = Net::HTTP.new(uri.host, uri.port)
http.use_ssl = false

req = Net::HTTP::Post.new(uri.path, {'Content-Type' =>'application/json',  'Authorization' => "Bearer #{token}"})

payload = []
n = 0

CSV.foreach(filename) do |row|
  payload << {
    "event" => row[2],
    "entity_type" => "user",
    "entity_id" => row[1],
    "target_entity_type" => "item",
    "target_entity_id" => row[3],
    "timestamp" => use_timestamp ? row[0].to_i : Time.now.getutc.to_i
  }
  n = n + 1

  if n == batch_size
    req.body =  payload.to_json
  
    start = DateTime.now.strftime('%Q').to_i  
    res = http.request(req)
    stop = DateTime.now.strftime('%Q').to_i

    puts "Code: #{res.code} - #{stop - start} ms."

    # reset
    n = 0
    payload = []
  end

end

# send the remaining records
if payload.size > 0
  req.body =  payload.to_json
      
  start = DateTime.now.strftime('%Q').to_i  
  res = http.request(req)
  stop = DateTime.now.strftime('%Q').to_i

  puts "Code: #{res.code} - #{stop - start} ms."

end