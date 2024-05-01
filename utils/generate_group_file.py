import cv2
import pandas as pd

from .util_myself import drawbox

X = lambda x, y, a, b, c : a*x + b*y + c
Y = lambda x, y, d, e, f : d*x + e*y + f

H_V = lambda x,y,j,k : "horizontal" if j-x > k-y else "vertical"

def read_label_xml_file( file_name ) :
  try:
    f = open( file_name + '.xml', 'r', encoding="utf8")
    for i in range(9) : # read garbage
      f.readline()
  except:
    f = open( file_name + '.xml', 'r', encoding='utf16')
    for i in range(9) : # read garbage
      f.readline()

  filename = f.readline().split("'")[1]

  top = []
  left = []
  width = []
  height = []
  id = []
  x = []
  y = []
  score = []
  label = []
  correction = []

  lines = f.readlines()

  for i in range( 0, len(lines) - 4, 4 ) :
    #print(lines[i])
    temp = lines[i].split("'")
    top.append( int(temp[1]) )
    left.append( int(temp[3]) )
    width.append( int(temp[5]) )
    height.append( int(temp[7]) )
    id.append( int(temp[9]) )
    x.append( (temp[11]) )
    y.append( (temp[13]) )
    score.append( float(temp[15]) )
    label.append( (lines[i+1].split(">")[1]).split("<")[0] )
    correction.append( (lines[i+2].split(">")[1]).split("<")[0] )


  f.close()
  return pd.DataFrame(list(zip( left,
                               top,
                               [left[i]+width[i] for i in range(len(left))],
                               [height[i]+top[i] for i in range(len(height))],
                               width,
                               height,
                               [width[i]*height[i] for i in range(len(height))],
                               id,
                               x,
                               y,
                               score,
                               label,
                               correction)),
                       columns =['top_x', 'top_y', 'buttom_x', 'buttom_y', 'width', 'height', 'area', 'id', 'x', 'y', 'score', 'label', 'correction'])

def generate_file( file_name, input_path, output_path, result_visualize  ) :
    file_name = file_name[:-4]
    xml_file = read_label_xml_file( output_path + "/ocr/" + file_name ) 

    group_id = sorted(list(set(xml_file['id'].values.tolist())))
    group_top_x = ((xml_file[['top_x','id']].groupby('id', group_keys=True).min()).sort_index())['top_x'].values.tolist()
    group_top_y = ((xml_file[['top_y','id']].groupby('id', group_keys=True).min()).sort_index())['top_y'].values.tolist()
    group_buttom_x = ((xml_file[['buttom_x','id']].groupby('id', group_keys=True).max()).sort_index())['buttom_x'].values.tolist()
    group_buttom_y = ((xml_file[['buttom_y','id']].groupby('id', group_keys=True).max()).sort_index())['buttom_y'].values.tolist()
    group_HorV = [ H_V(group_top_x[i], group_top_y[i], group_buttom_x[i], group_buttom_y[i]) for i in range(len(group_top_x))]
    group_x = []
    group_y = []

    convert_parameter = [] # [A,D,B,E,C,F]
    f = open( input_path + "/" + file_name + ".jgw", "r" )
    for line in f.readlines():
        convert_parameter.append((float(line)))
    f.close()

    for j in range(len(group_id)) :
        group_x.append( X( int((group_top_x[j]+group_buttom_x[j])/2), int((group_top_y[j]+group_buttom_y[j])/2), convert_parameter[0], convert_parameter[2], convert_parameter[4]) )
        group_y.append( Y( int((group_top_x[j]+group_buttom_x[j])/2), int((group_top_y[j]+group_buttom_y[j])/2), convert_parameter[1], convert_parameter[3], convert_parameter[5]) )

    all_group = pd.DataFrame(list(zip( group_id, group_top_x, group_top_y, group_buttom_x, group_buttom_y, group_HorV, group_x, group_y )), columns =['group_id', 'group_top_x', 'group_top_y', 'group_buttom_x', 'group_buttom_y', 'HorV', 'group_x', 'group_y'])

    f = open( output_path + "/ocr/" + file_name + "_group.txt", 'w', encoding='utf16' )

    group_list = xml_file[['label','id']].groupby('id', group_keys=True).groups
    place_name = []

    if result_visualize :
        image = cv2.imread( input_path + "/" + file_name + ".jpg" )
    for i in group_list :

        temp_place_name = ""
        if all_group.loc[all_group['group_id']== i ]['HorV'].tolist()[0] != 'vertical' :
            temp = (xml_file.loc[xml_file['id'] == i ]).sort_values(['x'], ascending = [False])
        else :
            temp = (xml_file.loc[xml_file['id'] == i ]).sort_values(['y'], ascending = [False])

        temp_place_name = ""
        for k in temp['label'] :
            temp_place_name+=k

        if temp_place_name != "" :
            place_name.append( temp_place_name )
        f.write( temp_place_name + '\t' + str(all_group.loc[all_group['group_id'] == i ]['group_x'].values[0]) + '\t' + str(all_group.loc[all_group['group_id'] == i ]['group_y'].values[0]) + "\n")

        if result_visualize :
            x1 = all_group.loc[all_group['group_id'] == i ]['group_top_x'].values[0]
            y1 = all_group.loc[all_group['group_id'] == i ]['group_top_y'].values[0]
            x2 = all_group.loc[all_group['group_id'] == i ]['group_buttom_x'].values[0]
            y2 = all_group.loc[all_group['group_id'] == i ]['group_top_y'].values[0]
            x3 = all_group.loc[all_group['group_id'] == i ]['group_buttom_x'].values[0]
            y3 = all_group.loc[all_group['group_id'] == i ]['group_buttom_y'].values[0]
            x4 = all_group.loc[all_group['group_id'] == i ]['group_top_x'].values[0]
            y4 = all_group.loc[all_group['group_id'] == i ]['group_buttom_y'].values[0]

            image = drawbox( image, x1, y1, x2, y2, x3, y3, x4, y4, (0,255,0), 1, True, temp_place_name )
    f.close()
    cv2.imwrite( output_path + "/ocr/" + file_name + "_result.jpg", image )