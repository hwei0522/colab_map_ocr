import math
import opencc
import openai
import numpy as np

from retrying import retry
from pyhull.delaunay import DelaunayTri

def group_word_by_thres_compress_2D_information( middle_axis_x, middle_axis_y, word_id, thres ):
  group_word = []
  while len( middle_axis_x ) > 0 :
    group = []
    group.append([middle_axis_x[0],middle_axis_y[0],word_id[0]])
    del middle_axis_x[0]
    del middle_axis_y[0]
    del word_id[0]
    num = 0
    while len( middle_axis_x ) > num :
      remove_item = False
      for i in group :
        if abs( middle_axis_x[num] - i[0] ) < thres :
            if abs( middle_axis_y[num] - i[1] ) < thres :
                if math.sqrt(( middle_axis_x[num] - i[0] ) ** 2 + ( middle_axis_y[num] - i[1] ) ** 2) < thres :
                  group.append([middle_axis_x[num],middle_axis_y[num],word_id[num]])
                  del middle_axis_x[num]
                  del middle_axis_y[num]
                  del word_id[num]
                  remove_item = True
        if remove_item : break
      if not remove_item : num+=1

    group_word.append(group)
  return group_word
  
def group_word_by_thres_keep_2D_information( middle_axis_x, middle_axis_y, word_id, thres ):
    
  group_word = []
  while len( middle_axis_x ) > 0 :
    group = []
    group.append([middle_axis_x[0],middle_axis_y[0],word_id[0]])
    del middle_axis_x[0]
    del middle_axis_y[0]
    del word_id[0]
    num = 0
    while len( middle_axis_x ) > num :
      remove_item = False
      for i in group :
        if abs( middle_axis_y[num] - i[1] ) < 20 :
          if abs( middle_axis_x[num] - i[0] ) < thres :
            group.append([middle_axis_x[num],middle_axis_y[num],word_id[num]])
            del middle_axis_x[num]
            del middle_axis_y[num]
            del word_id[num]
            remove_item = True
        if remove_item : 
          num = 0
          break
      if not remove_item : num+=1

    group_word.append(group)
  return group_word
 
def chatGPT_ckeck( location, predict ):
  converter = opencc.OpenCC('s2t.json')
  OPENAI_API_KEY = ''
  OPENAI_ORGNIIZATION = ""
  OPENAI_MODEL = "gpt-4"

  openai.api_key = OPENAI_API_KEY
  openai.organization = OPENAI_ORGNIIZATION

  # @retry(wait_fixed=5000, stop_max_attempt_number=50)
  def get_gpt_reply(prompt, input, temperature=0.25):
      completion = openai.ChatCompletion.create(
          model=OPENAI_MODEL,
          temperature=temperature,
          messages=[
              {"role": "user", "content": prompt+input}
          ]
      )
      return completion.get('choices')[0]['message']['content']

  promp_start = "想像你是研究「1930年中華民國地圖」的地理學家,你正在編地圖,這張地圖上有些確定的地理名稱,「"
  location = converter.convert( location )
  promp_middle = "」,你用了OCR工具,找到了以下地名「"
  promp_end = "」,請推敲是否可能為合理地理名稱，無須十分確定及專業知識，僅給予是否合理的建議即可，並在最後單獨以「結論：合理」或「結論：不合理」顯示"

  # print( promp_start + location + promp_middle + predict + promp_end )
  reply = get_gpt_reply( "", promp_start + location + promp_middle + predict + promp_end ) 
  # print( reply )
  if '結論：合理' in reply :
    return True
  else :
    return False

def angle_between(p1, p2) :
  return math.degrees(math.atan2((p1[1] - p2[1]), (p1[0] - p2[0])))

Distance = lambda x,y,j,k : math.sqrt(( x - j ) ** 2 + ( y - k ) ** 2 )

def find_near_by_angle( point_connect, middle_axis_list, finish_point, temp_group, group_distance, group_angle ) :
  target = temp_group[-1]
  if target in finish_point or target in temp_group[:-2] : return
  for i in point_connect[target] :
    if i in finish_point or i in temp_group : continue
    for j in temp_group :
      #print(i,j)
      #print(abs( distance( middle_axis_list[j][0], middle_axis_list[j][1], middle_axis_list[i][0], middle_axis_list[i][1] ) - group_distance ))
      if abs( Distance( middle_axis_list[j][0], middle_axis_list[j][1], middle_axis_list[i][0], middle_axis_list[i][1] ) - group_distance ) < 15 :
          degree = angle_between( (middle_axis_list[j][0], middle_axis_list[j][1]), (middle_axis_list[i][0], middle_axis_list[i][1]) )
          if degree > 180 : degree -= 180
          #print( abs( degree - group_angle ) )
          range_list = [group_angle + 10,group_angle - 10]
          range_list = [ k-180 if k > 180 else k for k in range_list ]
          range_list = [ k+180 if k < 0 else k for k in range_list ]
          #print(range_list)
          if range_list[0] > range_list[1] :
            if degree < range_list[0] and degree > range_list[1]:
              temp_group.append(i)
              find_near_by_angle( point_connect, middle_axis_list, finish_point, temp_group, group_distance, group_angle )
              break
          else :
            if degree < range_list[0] or degree > range_list[1]:
              temp_group.append(i)
              find_near_by_angle( point_connect, middle_axis_list, finish_point, temp_group, group_distance, group_angle )
              break

def group_place_name( x_min, y_min, x_max, y_max ) :

  large = [[],[],[],[]] # middle_axis_x, middle_axis_y ,id, width+length
  small = [[],[],[],[]] # middle_axis_x, middle_axis_y ,id, width+length
  final_group = []

  for i in range( len(x_min) ) :
    if (x_max[i]-x_min[i])*(y_max[i]-y_min[i]) > 3000 :
        large[0].append(int((x_min[i]+x_max[i])/2))
        large[1].append(int((y_min[i]+y_max[i])/2))
        large[2].append(i)
        large[3].append(abs(x_max[i]-x_min[i]))
        large[3].append(abs(y_max[i]-y_min[i]))
    else :
        small[0].append(int((x_min[i]+x_max[i])/2))
        small[1].append(int((y_min[i]+y_max[i])/2))
        small[2].append(i)
        small[3].append(abs(x_max[i]-x_min[i]))
        small[3].append(abs(y_max[i]-y_min[i]))

  for k in [large,small] : # large & small
      middle_axis_x = []
      middle_axis_y = []
      word_id = []
      

      middle_axis_list = [ [k[0][i], k[1][i]] for i in range( len(k[0]) ) ]
      max_word_size = max( k[3] )

      if len(middle_axis_list) == 1 :
        final_group.append( [k[2][0]] )
        continue


      tri = DelaunayTri( middle_axis_list )
      DT = tri.vertices
      point_connect = [ [] for i in range( len(middle_axis_list) ) ]
      for i in DT :
        point_connect[i[0]].append(i[1])
        point_connect[i[0]].append(i[2])
        point_connect[i[1]].append(i[0])
        point_connect[i[1]].append(i[2])
        point_connect[i[2]].append(i[1])
        point_connect[i[2]].append(i[0])
      for i in range( len(point_connect) ) :
        point_connect[i] = list(set(point_connect[i]))

      finish_point = []
      num_i = 0
      while num_i < len( middle_axis_list ) :
        #print(finish_point)
        if num_i in finish_point :
          num_i += 1
          continue
        temp_group = [num_i]
        temp_group_degree = -1
        temp_group_distance = -1

        for i in point_connect[num_i] :
          if i in finish_point or i in temp_group : continue

          if len( temp_group ) == 1 :
            if Distance( middle_axis_list[num_i][0], middle_axis_list[num_i][1], middle_axis_list[i][0], middle_axis_list[i][1] ) < 1.2 * max_word_size :
              temp_group.append(i)
              temp_group_degree = angle_between( (middle_axis_list[num_i][0], middle_axis_list[num_i][1]), (middle_axis_list[i][0], middle_axis_list[i][1]) )
              temp_group_distance = Distance( middle_axis_list[num_i][0], middle_axis_list[num_i][1], middle_axis_list[i][0], middle_axis_list[i][1] )
              if temp_group_degree > 180 : temp_group_degree -= 180
              #print(temp_group_degree,temp_group_distance)
              find_near_by_angle( point_connect, middle_axis_list, finish_point, temp_group, temp_group_distance, temp_group_degree )
          else :
            for j in temp_group :
              if abs( Distance( middle_axis_list[j][0], middle_axis_list[j][1], middle_axis_list[i][0], middle_axis_list[i][1] ) - temp_group_distance ) < 15 :
                degree = angle_between( (middle_axis_list[j][0], middle_axis_list[j][1]), (middle_axis_list[i][0], middle_axis_list[i][1]) )
                if degree > 180 : degree -= 180
                range_list = [ temp_group_degree + 10,temp_group_degree - 10 ]
                range_list = [ k-180 if k > 180 else k for k in range_list ]
                range_list = [ k+180 if k < 0 else k for k in range_list ]
                if range_list[0] > range_list[1] :
                  if degree < range_list[0] and degree > range_list[1]:
                    temp_group.append(i)
                    find_near_by_angle( point_connect, middle_axis_list, finish_point, temp_group, temp_group_distance, temp_group_degree )
                else :
                  if degree < range_list[0] or degree > range_list[1]:
                    temp_group.append(i)
                    find_near_by_angle( point_connect, middle_axis_list, finish_point, temp_group, temp_group_distance, temp_group_degree )

        if len( temp_group ) >= 2:
          temp = []
          for i in range(len(temp_group)) :
            finish_point.append(temp_group[i])
            temp.append( k[2][temp_group[i]] )
          final_group.append( temp )
        else :
          middle_axis_x.append(k[0][temp_group[0]])
          middle_axis_y.append(k[1][temp_group[0]])
          word_id.append(k[2][temp_group[0]])

        num_i += 1

      thres = 50
      while len( middle_axis_x ) > 0 :
        if len( middle_axis_x ) == 1 :
          final_group.append([word_id[0]])
          break

        temp_group = group_word_by_thres_compress_2D_information( middle_axis_x, middle_axis_y, word_id, thres )
        #print(temp_group)
        for i in temp_group :
          #print(i)
          if len(i) >= 2 :
            temp = []
            for j in i :
              temp.append(j[2])
            final_group.append(temp)
          else :
            middle_axis_x.append(i[0][0])
            middle_axis_y.append(i[0][1])
            word_id.append(i[0][2])
        #print( "-----")
        thres += 30

  group_list = [0]*len(x_min)
  num = 0
  for i in final_group :
    for j in i :
      group_list[j] = num
    num += 1

  return group_list, num+1


  '''
  large = [[],[],[]] # middle_axis_x, middle_axis_y ,id
  small = [[],[],[]] # middle_axis_x, middle_axis_y ,id
  final_group = []

  for i in range( len(x_min) ) :
    if (x_max[i]-x_min[i])*(y_max[i]-y_min[i]) > 3000 :
        large[0].append(int((x_min[i]+x_max[i])/2))
        large[1].append(int((y_min[i]+y_max[i])/2))
        large[2].append(i)
    else :
        small[0].append(int((x_min[i]+x_max[i])/2))
        small[1].append(int((y_min[i]+y_max[i])/2))
        small[2].append(i)

  for k in [large,small] : # large & small
      middle_axis_x = k[0]
      middle_axis_y = k[1]
      word_id = k[2]
      
      thres = 50
      round_num = 1
      while round_num <= 3 :
        
        if len( middle_axis_x ) == 1 :
          final_group.append([[middle_axis_x[0],middle_axis_y[0],word_id[0]]])
          break
      
        # horizontal
        temp_group = group_word_by_thres_keep_2D_information( middle_axis_x, middle_axis_y, word_id, thres )
        for i in temp_group :
          if len(i) >= 2 :
            final_group.append(i)
          else :
            middle_axis_x.append(i[0][0])
            middle_axis_y.append(i[0][1])
            word_id.append(i[0][2])
        
        # vertical
        temp = middle_axis_x
        middle_axis_x = middle_axis_y
        middle_axis_y = temp
        temp_group = group_word_by_thres_keep_2D_information( middle_axis_x, middle_axis_y, word_id, thres )
        for i in temp_group :
          if len(i) >= 2 :
            final_group.append(i)
          else :
            middle_axis_x.append(i[0][0])
            middle_axis_y.append(i[0][1])
            word_id.append(i[0][2])
        
        temp = middle_axis_x
        middle_axis_x = middle_axis_y
        middle_axis_y = temp
        thres += 25
        
        round_num += 1
      
      
      thres = 50
      while len( middle_axis_x ) > 0 :
        if len( middle_axis_x ) == 1 :
          final_group.append([[middle_axis_x[0],middle_axis_y[0],word_id[0]]])
          break

        temp_group = group_word_by_thres_compress_2D_information( middle_axis_x, middle_axis_y, word_id, thres )
        for i in temp_group :
          if len(i) >= 2 :
            final_group.append(i)
          else :
            middle_axis_x.append(i[0][0])
            middle_axis_y.append(i[0][1])
            word_id.append(i[0][2])
        
        thres += 30
        
  group_list = [0]*len(x_min)
  num = 0
  for i in final_group :
      for j in i :
          group_list[j[2]] = num
      num += 1

  return group_list
  '''