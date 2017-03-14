import time
import sys
import os
import numpy
import gzip
import zipfile
import cPickle
import random
import PIL.Image

try:
  import magic
  ms = magic.open(magic.MAGIC_NONE)
  ms.load()
except ImportError: # no magic module
  ms = None

class fx_UnpickleError(Exception):
  pass

def fx_pickle(filename, data, compress=False):
  if compress:
    fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
    fo.writestr('data', cPickle.dumps(data, -1))
  else:
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
  fo.close()
    
def fx_unpickle(filename):
  if not os.path.exists(filename):
    raise fx_UnpickleError("Path '%s' does not exist." % filename)
  if ms is not None and ms.file(filename).startswith('gzip'):
    fo = gzip.open(filename, 'rb')
    dict = cPickle.load(fo)
  elif ms is not None and ms.file(filename).startswith('Zip'):
    fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
    dict = cPickle.loads(fo.read('data'))
  else:
    fo = open(filename, 'rb')
    dict = cPickle.load(fo)
  
  fo.close()
  return dict

def fx_squre_distant(p, q, pSOS=None, qSOS=None):
  if pSOS is None:
    pSOS = (p**2).sum(axis=1)
    qSOS = (q**2).sum(axis=1)

  return pSOS.reshape(-1,1) + qSOS - 2 * numpy.dot(p, q.T)
  
def fx_cos_distant(p, q):
  dist = numpy.dot(p, q.T) / numpy.sqrt((p ** 2).sum(axis=1)).reshape((p.shape[0],1))
  return dist / numpy.sqrt((q ** 2).sum(axis=1))
  
def fx_distant(p, q, type='L1'):
  if type == 'L1':
    prows, pcols = p.shape
    qrows, qcols = q.shape
    res = numpy.zeros((prows, qrows))
    for i in range(qrows):
      res[:,i] = (numpy.abs(p - q[i,:])).sum(axis=1)
    return res
    
def fx_scale_to_unit_interval(ndar, eps=1e-8):
  ''' Scales all values in the ndarray ndar to be between 0 and 1 '''
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar

def fx_tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  '''
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`PIL.Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  '''

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape    = [0,0]
  # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
    assert len(X) == 4
    # Create an output numpy ndarray to store the image
    if output_pixel_vals:
      out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                              dtype='uint8')
    else:
      out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                              dtype=X.dtype)

    #colors default to 0, alpha defaults to 1 (opaque)
    if output_pixel_vals:
      channel_defaults = [0, 0, 0, 255]
    else:
      channel_defaults = [0., 0., 0., 1.]

    for i in xrange(4):
      if X[i] is None:
        # if channel is None, fill it with zeros of the correct
        # dtype
        dt = out_array.dtype
        if output_pixel_vals:
          dt = 'uint8'
        out_array[:, :, i] = numpy.zeros(out_shape,
              dtype=dt) + channel_defaults[i]
      else:
        # use a recurrent call to compute the channel and store it
        # in the output
        out_array[:, :, i] = tile_raster_images(
            X[i], img_shape, tile_shape, tile_spacing,
            scale_rows_to_unit_interval, output_pixel_vals)
    return out_array

  else:
    # if we are dealing with only one channel
    H, W = img_shape
    Hs, Ws = tile_spacing

    # generate a matrix to store the output
    dt = X.dtype
    if output_pixel_vals:
      dt = 'uint8'
    out_array = numpy.zeros(out_shape, dtype=dt)

    for tile_row in xrange(tile_shape[0]):
      for tile_col in xrange(tile_shape[1]):
        if tile_row * tile_shape[1] + tile_col < X.shape[0]:
          this_x = X[tile_row * tile_shape[1] + tile_col]
          
          if scale_rows_to_unit_interval:
            # if we should scale values to be between 0 and 1
            # do this by calling the `scale_to_unit_interval`
            # function
            this_img = fx_scale_to_unit_interval(
                this_x.reshape(img_shape))
          else:
            this_img = this_x.reshape(img_shape)
          # add the slice to the corresponding position in the
          # output array
          c = 1
          if output_pixel_vals:
            c = 255
          out_array[
              tile_row * (H + Hs): tile_row * (H + Hs) + H,
              tile_col * (W + Ws): tile_col * (W + Ws) + W
              ] = this_img * c
      return out_array
      