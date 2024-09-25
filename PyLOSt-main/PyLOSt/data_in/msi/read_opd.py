# coding=utf-8
def Read_opd_file(fileName, showImg=False):
    """
    The function is written in python 2.7, in window7.
    The function is based on a matlab code file: READOPD.
    The function can read a OPD file and show us an image, which is a picture(of a piece of miroi, for example)
    If you wanna run this code, please check that you have well installed the numpy, matplotlib modules

    :param fileName: opd file path
    :param showImg: Display raw image
    :return: Raw data as dictionary
    """

    import struct
    import numpy as np
    from matplotlib import pyplot as plt

    d = {}

    fid = open(fileName, "rb")

    # initial the value
    scaleByWavelength = 0  # if set to 1, will scale the opd by the wavelength
    fileIndex = 0
    badpixelValue = np.nan
    rotate = 1

    # We read the data from the beginning of the file
    fid.seek(0)
    signature_general_decode = fid.read(2)
    name_general_decode = fid.read(16)
    type_general_decode = fid.read(2)
    len_general_decode = fid.read(4)
    attr_general_decode = fid.read(2)

    signature_general = struct.unpack("<2c", signature_general_decode)[
        0]  # unpack the binary data to get the value of the signature
    name_general = struct.unpack("<16s", name_general_decode)
    name_general = str(name_general).split('\\x00')  # Remove Extra Whitespace
    name_general = name_general[
        0]  # The result of unpacking is a tuple, what we want is just the first element of the tuple.
    name_general = str(name_general)[2:]  # "Remove unmeaningful symbols, ( and ".
    type_general = struct.unpack("<h", type_general_decode)[0]  # unpack the binary data to get the value of the name
    len_general = struct.unpack("<l", len_general_decode)[0]  # unpack the binary data to get the value of the length
    attr_general = struct.unpack("<H", attr_general_decode)[
        0]  # unpack the binary data to get the value of the attribute

    BLCK_SIZE = 24  # in every row of the second part(also block part),it has 24 bytes.(16+2+4+2=24)
    BPF = float('1e+38')
    num_blcks = int(len_general / BLCK_SIZE)  # number of row
    counter = 1
    mult_offset = 0
    offset = 0

    # read the block
    block_name = []
    block_type = []
    block_len = []
    block_attr = []

    for i in range(0, (num_blcks - 1)):
        name_encode = fid.read(16)
        type_encode = fid.read(2)
        len_encode = fid.read(4)
        attr_encode = fid.read(2)

        name_decode = struct.unpack("<16s", name_encode)
        name_decode = str(name_decode).split('\\x00')
        name_decode = name_decode[0]
        name_decode = str(name_decode)[3:]
        type_decode = struct.unpack("<h", type_encode)[0]
        len_decode = struct.unpack("<l", len_encode)[0]
        attr_decode = struct.unpack("<H", attr_encode)[0]

        block_name.append(name_decode)
        block_type.append(type_decode)
        block_len.append(len_decode)
        block_attr.append(attr_decode)

        if len_decode > 0:
            counter = counter + 1

    phaseSize = 4
    mult_decode = 1
    print(fid.tell())

    # Ususlly we have read 6002 bytes right here.

    # read the data from data blocks
    for i in range(0,
                   249):  # Acuually there are 250 rows in the second part(block part), but the first row is for itself,
        # for axample, the length value in the first row means the length of the block part(so there id no data correspondent in the third part)
        # so to read the third part, we just use the rest 249 rows, so we have i in range(0,249), not range(0,250)
        if block_len[i] > 0:
            if block_name[i] == "RAW DATA":
                print("RAW DATA %d" % (fid.tell()))
                rows_encode = fid.read(2)
                rows = struct.unpack("<H", rows_encode)[0]
                cols_encode = fid.read(2)
                cols = struct.unpack("<H", cols_encode)[0]
                elsize_encode = fid.read(2)
                elsize = struct.unpack("<h", elsize_encode)[0]

                if elsize == 4:
                    format = "f"
                    way_decode = "<" + str(cols * rows) + format
                elif elsize == 2:
                    format = "h"
                    way_decode = "<" + str(cols * rows) + format
                else:
                    format = "c"
                    way_decode = "<" + str(cols * rows) + format
                if (cols * rows) > (1000 * 1000):
                    fileIndex = fid.tell()
                    rows1 = rows
                    cols1 = cols
                    way1_decode = way_decode
                else:
                    arrayTmp_encode = fid.read(cols * rows * elsize)
                    arrayTmp = struct.unpack(way_decode, arrayTmp_encode)[0]
                    arrayTmp = np.asarray(arrayTmp)

            elif block_name[i] == "RAW_DATA":
                print("RAW_DATA %d" % (fid.tell()))
                rows_encode = fid.read(2)
                rows = struct.unpack("<H", rows_encode)[0]
                cols_encode = fid.read(2)
                cols = struct.unpack("<H", cols_encode)[0]

                elsize = fid.read(2)
                elsize = struct.unpack("<h", elsize)[0]
                if elsize == 4:
                    prec = "float"
                    n = 4
                    format = "f"
                    way_decode = "<" + str(cols * rows) + format
                elif elsize == 2:
                    prec = "short"
                    n = 2
                    format = "h"
                    way_decode = "<" + str(cols * rows) + format
                else:
                    prec = "char"
                    n = 1
                    format = "c"
                    way_decode = "<" + str(cols * rows) + format

                if (cols * rows) > (1000 * 1000):
                    fileIndex = fid.tell()
                    rows1 = rows
                    cols1 = cols
                    way1_decode = way_decode
                else:
                    arrayTmp_encode = fid.read(rows * cols * elsize)
                    arrayTmp = struct.unpack(way_decode, arrayTmp_encode)
                    arrayTmp = np.asarray(arrayTmp)

            elif block_name[i] == "SAMPLE_DATA":
                print("SAMPLE_DATA %d" % (fid.tell()))
                rows_encode = fid.read(2)
                rows = struct.unpack("<H", rows_encode)
                cols_encode = fid.read(2)
                cols = struct.unpack("<H", cols_encode)

                elsize = fid.read(2)
                elsize = struct.unpack("<h", elsize)[0]
                if elsize == 4:
                    format = "f"
                    way_decode = "<" + str(cols * rows) + format
                elif elsize == 2:
                    format = "h"
                    way_decode = "<" + str(cols * rows) + format
                else:
                    format = "c"
                    way_decode = "<" + str(cols * rows) + format
                if (cols * rows) > (1000 * 1000):
                    fileIndex = fid.tell()
                    rows1 = rows
                    cols1 = cols
                    way1_decode = way_decode
                else:
                    arrayTmp_encode = fid.read(rows * cols * elsize)
                    arrayTmp = struct.unpack(way_decode, arrayTmp_encode)
                    arrayTmp = np.asarray(arrayTmp)

            elif block_name[i] == "OPD":
                print("OPD %d" % (fid.tell()))
                rows_encode = fid.read(2)
                rows = struct.unpack("<H", rows_encode)[0]
                cols_encode = fid.read(2)
                cols = struct.unpack("<H", cols_encode)[0]
                elsize = fid.read(2)
                elsize = struct.unpack("<h", elsize)[0]
                if elsize == 4:
                    format = "f"
                    way_decode = "<" + str(cols * rows) + format
                elif elsize == 2:
                    format = "h"
                    way_decode = "<" + str(cols * rows) + format
                else:
                    format = "c"
                    way_decode = "<" + str(cols * rows) + format

                if (cols * rows) > (1000 * 1000):
                    fileIndex = fid.tell()
                    rows1 = rows
                    cols1 = cols
                    way1_decode = way_decode
                else:
                    arrayTmp_encode = fid.read(rows * cols * elsize)
                    arrayTmp = struct.unpack(way_decode, arrayTmp_encode)
                    for i in range(0, (cols * rows - 1)):
                        arrayTmp[i] = round(arrayTmp[i], 5)
                    arrayTmp = np.asarray(arrayTmp)

            elif block_name[i] == "Image":
                # print ("OPD %d" % (fid.tell()))
                rows_image_encode = fid.read(2)
                rows_image = struct.unpack("<H", rows_image_encode)[0]
                cols_image_encode = fid.read(2)
                cols_image = struct.unpack("<H", cols_image_encode)[0]
                elsize_image = fid.read(2)
                elsize_image = struct.unpack("<h", elsize_image)[0]
                if elsize == 4:
                    format = "f"
                    way_image_decode = "<" + str(cols_image * rows_image) + format
                elif elsize == 2:
                    format = "h"
                    way_image_decode = "<" + str(cols_image * rows_image) + format
                else:
                    format = "c"
                    way_image_decode = "<" + str(cols_image * rows_image) + format

                arrayTmp_image_encode = fid.read(rows_image * cols_image * elsize_image)
                arrayTmp_image = struct.unpack("<307200c", arrayTmp_image_encode)

                # print "the length of image is "+str(len(arrayTmp_image_encode))
                # print "the rows of image is "+str(rows_image)
                # print "the cols of image is "+str(cols_image)
                # print "the elsize of image is "+str(elsize_image)

            elif block_name[i] == "Wavelength":
                wavelength_encode = fid.read(4)
                wavelength_decode = struct.unpack("<f", wavelength_encode)[0]
                d[block_name[i]] = wavelength_decode

            elif block_name[i] == "StageX":
                stagex_encode = fid.read(4)
                stagex_decode = struct.unpack("<f", stagex_encode)[0]
                d[block_name[i]] = stagex_decode * 25.4  # inches to mm

            elif block_name[i] == "StageY":
                stagey_encode = fid.read(4)
                stagey_decode = struct.unpack("<f", stagey_encode)[0]
                d[block_name[i]] = stagey_decode * 25.4  # inches to mm

            elif block_name[i] == "Mult":
                mult_encode = fid.read(2)
                mult_decode = struct.unpack("<h", mult_encode)[0]

            elif block_name[i] == "Aspect":
                aspect_encode = fid.read(4)
                aspect_decode = struct.unpack("<f", aspect_encode)[0]

            elif block_name[i] == "Pixel_size":
                pixel_size_encode = fid.read(4)
                pixel_size_decode = struct.unpack("<f", pixel_size_encode)[0]
                d[block_name[i]] = pixel_size_decode

            elif block_name[i] == "Date":
                date_encode = fid.read(10)
                d[block_name[i]] = date_encode

            elif block_type[i] == 12:
                fid.seek(block_len[i], 1)  # long
            elif block_type[i] == 8:
                fid.seek(block_len[i], 1)  # double

            elif block_type[i] == 7:
                fid.seek(block_len[i], 1)  # float

            elif block_type[i] == 6:
                fid.seek(block_len[i], 1)  # short

            elif block_type[i] == 5:
                fid.seek(block_len[i], 1)  # string

            elif block_type[i] == 3:
                rows_encode = fid.read(2)
                rows = struct.unpack("<H", rows_encode)[0]
                cols_encode = fid.read(2)
                cols = struct.unpack("<H", cols_encode)[0]
                elsize_encode = fid.read(2)
                elsize = struct.unpack("<h", elsize_encode)[0]
                fid.seek(rows * cols * elsize, 1)
                d['rows'] = rows
                d['cols'] = cols
            else:
                pass
        else:
            pass

    arrayTmp = np.reshape(arrayTmp, (640, 480))
    print(np.size(arrayTmp))
    arrayTmp = np.transpose(arrayTmp)

    if scaleByWavelength == 1:
        scale = wavelength_decode / mult_decode
    else:
        scale = 1 / mult_decode

    if phaseSize == 2:
        badValue = 32766
    else:
        badValue = BPF

    if fileIndex > 0:
        # we will read a col at a time
        if fid.seek(fileIndex, 0) == 0:
            # create our array
            array = np.ones(cols1, rows1) * badpixelValue
            for i in range(rows1):
                col_encode = fid.read(cols1 * elsize)
                col = struct.unpack(way1_decode, col_encode)
                col_array = np.asarray(col)
                col_array_matrix_transpose = np.asmatrix(col_array)
                col_array_matrix = np.transpose(col_array_matrix_transpose)
                col = np.flipud(col_array_matrix)
                col = np.asarray(col)

                indexGood = np.where(col < badValue)
                array[indexGood, i] = col[indexGood] * scale
    else:

        for i in range(0, 480):
            for j in range(0, 640):
                if arrayTmp[i][j] > badValue:
                    arrayTmp[i][j] = badpixelValue
                else:
                    arrayTmp[i][j] = (arrayTmp[i][j]) * scale

    array = np.flipud(arrayTmp)
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    array_mean = np.nanmean(array)
    array_relative = array  # (array-array_mean)/(array_max-array_min)
    height_relative = array_relative * wavelength_decode
    d['height'] = height_relative

    # The peak-to-valley difference calculated over the entire measured array.
    Rt = (np.nanmax(arrayTmp) - np.nanmin(arrayTmp)) * wavelength_decode

    if showImg:
        # The root-mean-squared roughness calculated over the entire measured array.
        z = 0
        n = 0
        arrayTmp_relative = arrayTmp - np.nanmean(arrayTmp)
        for i in range(480):
            for j in range(640):
                if arrayTmp_relative[i][j] > 0:
                    z = z + np.square(arrayTmp_relative[i][j])
                    n = n + 1

                elif arrayTmp_relative[i][j] < 0:
                    z = z + np.square(arrayTmp_relative[i][j])
                    n = n + 1
                elif arrayTmp_relative[i][j] == 0:
                    z = z + np.square(arrayTmp_relative[i][j])
                    n = n + 1
                else:
                    pass

        # Get the image
        range_image = (0.9 * np.nanmean(height_relative) + 0.1 * np.nanmin(height_relative)), np.nanmax(height_relative)
        plt.imshow(height_relative, interpolation='nearest', clim=range_image)
        plt.title(fileName[-48:])
        plt.text(-20, -50, "Rq= " + str(np.sqrt(z / n) * wavelength_decode))
        plt.text(-20, -70, "Rt= " + str((np.nanmax(arrayTmp) - np.nanmin(arrayTmp)) * wavelength_decode))
        plt.text(-20, -115, "stage_x = " + str(stagex_decode * 25.4))
        plt.text(-20, -95, "stage_y = " + str(stagey_decode * 25.4))
        plt.colorbar()
        plt.show()

    fid.close()
    return d
