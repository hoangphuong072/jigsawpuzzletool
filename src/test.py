import math
import random

import numpy as np

# Danh sách các ma trận
original_pieces_array = [
    np.array([[1]]),  # Square 1 Piece
    np.array([[1, 1],
              [1, 1]]),  # Square 2 Piece
    np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]),  # Square 3 Piece

    np.array([[1, 1]]),  # line 2 Piece
    np.array([[1, 1, 1]]),  # line 3 Piece
    np.array([[1, 1, 1, 1]]),  # line 4 Piece
    np.array([[1, 1, 1, 1, 1]]),  # line 5 Piece

    np.array([[1, 0],
              [1, 1]]),  # L-Shaped 2 Piece
    np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 1, 1]]),  # L-Shaped 3 Piece
    np.array([[1, 0],
              [1, 1],
              [1, 0]]),
    np.array([[1, 0],
              [1, 0],
              [1, 1]]),
    np.array([[1, 0,1],
              [1, 1,1]]),
    np.array([[1, 0,0],
              [1, 1,0],
              [0, 1,1]]),
    np.array([[1, 0],
              [1, 1],
              [0, 1]]),
]

pieces_array = []
for piece in original_pieces_array:
    last_piece = np.array([0])
    for i in range(0, 4):
        rotate_piece = np.rot90(piece, i)
        piece_existed = False
        for added_piece in pieces_array:
            if np.array_equal(rotate_piece, added_piece):
                piece_existed = True
                break
        if piece_existed:
            continue
        last_piece = np.zeros(rotate_piece.shape)
        np.copyto(last_piece,rotate_piece)
        # print(last_piece)
        pieces_array.append(rotate_piece)

pieces_array = [(np.sum(piece), piece) for i, piece in enumerate(pieces_array)]
pieces_array.sort(key=lambda x: x[0])

# for piece in pieces_array:
#     print("----------------")
#     print(piece)

# exit(0)

max_row, max_col = 5, 10


# max_row += 2
# max_col += 2


def expand_matrix(original_matrix, position):
    original_rows, original_cols = len(original_matrix), len(original_matrix[0])

    expanded_matrix = np.zeros((max_row, max_col), dtype=int)
    row_start, col_start = position

    if row_start < 0 or col_start < 0 or row_start + original_rows > max_row or col_start + original_cols > max_col:
        return None

    expanded_matrix[row_start:row_start + original_rows, col_start:col_start + original_cols] = original_matrix

    return expanded_matrix


def check_available_piece(original_matrix, piece):
    result = original_matrix + piece
    if check_any_equal(result, 2):
        return False
    return True


def get_list_available_piece(original, position):
    list = []
    for sum, piece in pieces_array:
        expand_piece = expand_matrix(piece, position)
        if expand_piece is not None:
            if check_available_piece(original, expand_piece):
                list.append((sum, piece, expand_piece))

    return list


def get_random_index_piece(pieces):
    item = random.choices(pieces, [sum*sum for sum, p, ep in pieces], k=1)[0]
    return item[1], item[2]


def find_first_zero(_pieces_array):


    arr  = np.zeros(_pieces_array.size,dtype=[("x", int), ("y",int), ("value", float)])
    has_empty = False
    for (x, y), value in np.ndenumerate(_pieces_array):
        if value == 0:
            return (x, y)
            number = abs((max_row/2-(x))/(max_row/2))+abs((max_col/2-(y))/(max_col/2))
            arr[x*max_col+y] = (x,y,math.floor(number*10)/10)
            has_empty = True
        else:
            arr[x*max_col+y] = (x,y,100)

    if not has_empty:
        return None
    arr  = sorted(arr, key=lambda x: x[2])
    # print((arr))

    return (arr[0][0],arr[0][1])


def check_all_equal_one(_pieces_array):
    for (x, y), value in np.ndenumerate(_pieces_array):
        if value != 1:
            return False
    return True


def check_any_equal(_pieces_array, _value):
    for (x, y), value in np.ndenumerate(_pieces_array):
        if value == _value:
            return True
    return False


def print_grid(grid):
    s = ""
    for row in grid:
        for e in row:
            s+=" "+str(e)
        s+="\n"
    print(s)


puzzle = np.zeros((max_row, max_col), dtype=int)
#
# listPattern = []
# while not check_all_equal_one(puzzle):
#     # for _ in range(10):
#     print("step %d")
#     print_grid(puzzle)
#
#     first_zero_piece = find_first_zero(puzzle)
#     print(first_zero_piece)
#
#     list = get_list_available_piece(puzzle, first_zero_piece)
#     # for piece in list:
#     #     print("-------")
#     #     print(piece)
#     piece, expand_piece = get_random_index_piece(list)
#     listPattern.append(piece)
#     puzzle += expand_piece
#
#     print_grid(piece)
#     print_grid(puzzle)
#     print("------finished------")
#
# print("Puzzle finished")
#
# print_grid(puzzle)
# print("--------------------------------")
# print("Số lượng mảnh : " + str(len(listPattern)))
# print("--------------------------------")
# for piece in listPattern:
#     print("-----")
#     print_grid(piece)
# # break


arr = []
def find_combinations(target, pieces, current_combination,position):
    # Nếu tổng bằng target, in ra kết quả
    if check_all_equal_one(target):
        arr.append(current_combination)
        print(current_combination)
        return

    # Nếu target nhỏ hơn 0 hoặc không còn số nào để xem xét, dừng đệ quy
    if check_any_equal(target,2) or len(pieces) == 0 :
        return

    # Xét tất cả các trường hợp
    for piece in pieces:
        new_puzzle = np.zeros(puzzle.shape)
        np.copyto(new_puzzle,puzzle)
        print("piece")
        print(piece)
        new_puzzle += expand_matrix(piece[1],position)
        new_combination = current_combination.copy()
        new_combination.append(piece)
        find_combinations(new_puzzle, piece, new_combination,position)



# Gọi hàm để tìm các tổ hợp
# find_combinations(puzzle, pieces_array, [],(0,0))
# print(len(arr))


for row in range(0,max_row-1):
    for col in range(0,max_col-1):
        if puzzle[row,col]==0:
            for piece in pieces_array:
                puzzle +=expand_matrix(piece,(row,col))
                if puzzle
