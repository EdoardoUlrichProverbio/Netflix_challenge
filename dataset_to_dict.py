#MODULES

#Utility
import numpy as np

#----------------------------------------------------------------------------------

from collections import OrderedDict
def list_from_lines(list1):
  users = []
  movies = []
  temp = []
  movie = int(list1[0].replace(":\n", ""))-1
  ratings = []
  numberOfMovies = 1

  for line in list1[:100000]:
      if ":" in line and line != "1:\n":
          for x in range(len(temp)):
              movies.append(movie)
          temp = []
          movie = int(line.replace(":\n", ""))-1
          numberOfMovies += 1
      elif line != "1:\n":
          temp.append(line.split(",")[0])
          users.append(line.split(",")[0])
          ratings.append(int(line.split(",")[1]))
  if len(temp) != 0:
      for x in range(len(temp)):
          movies.append(movie)

  uniqueUsers = list(OrderedDict.fromkeys(users))

  dict = {}
  index = 0
  for element in uniqueUsers:
      dict.update({element: index})
      index += 1

  usersIndex = []
  for element in users:
      usersIndex.append(dict[element])
  return uniqueUsers, usersIndex, movies, ratings, numberOfMovies

#----------------------------------------------------------------------------------

def dict_from_lines(list1):

  dict1 = {}
  #for line in list1:
  for line in list1[:100000]:

    if ':' in line:
      movie_index = line.split(":")[0]
      dict1[movie_index] = []
    else :  dict1[movie_index].append(line.split(',')[:2])

  # list1 = None #free memory

  return dict1

#----------------------------------------------------------------------------------

def dict_to_ID_list(dict1):

  User_ID_list = []
  Movie_ID_list = []
  movies, users, ratings = [], [], []

  for key in list(dict1.keys()):

    Movie_ID_list.append(int(key))
    for coord in dict1[key]:
      User_ID_list.append(int(coord[0]))
      movies.append(coord[0])
      users.append(int(key))
      ratings.append(int(coord[1]))

    del dict1[key] #free memory

  User_ID_list = np.sort(np.unique(np.asarray(User_ID_list)))
  Movie_ID_list = np.asarray(Movie_ID_list)

  return User_ID_list, Movie_ID_list, ratings, users, movies