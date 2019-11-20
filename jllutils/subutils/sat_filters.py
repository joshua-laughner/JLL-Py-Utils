from abc import abstractmethod
import numpy as np
from numpy import ma


class FilteringError(Exception):
    pass


class SatelliteFilter(object):
    """
    Base class for all satellite data quality filters to use with :class:`SatelliteData`

    To create a filter for a dataset, subclass this class and implement the :meth:`get_mask` method. This method must
    take a handle to the satellite dataset and return a boolean array that is ``True`` for bad values.

    This should be sufficient for many cases, however for some datasets you may also need to override
    :meth:`broadcast_mask`. This method takes the mask returned by :meth:`get_mask` and expands it to the shape of the
    data array to be filtered. The base implementation assumes that the leading dimensions of the mask and the array
    to be filtered match. That is, a 500-by-8 mask could be applied to a 500-by-8 array, 500-by-8-by-n array, or any
    array so long as the first two dimensions are 500-by-8. If your data does not follow this pattern, then a more
    complicated broadcasting method will be needed.
    """
    def __init__(self, masked=False, fill=np.nan):
        """
        Instantiate the filter.

        :param masked: controls whether the filter converts the data array into a masked array and masks bad quality
         data (``True``) or fills it with a fill value (``False``, default).
        :type masked: bool

        :param fill: fill value to use if ``masked`` is ``False``.
        """
        self._use_masked_array = masked
        self._fill = fill

    @abstractmethod
    def get_mask(self, dataset):
        """
        Method that extracts a mask of good and bad values from the dataset

        :param dataset: handle to the dataset to get the mask for

        :return: the mask as a boolean array
        :rtype: :class:`numpy.ndarray`
        """
        pass

    @staticmethod
    def broadcast_mask(mask, shape):
        """
        Expand the mask to the shape of the array to be filtered
        :param mask: the boolean mask
        :type mask: :class:`numpy.ndarray`

        :param shape: shape of the array to be filtered
        :type shape: tuple

        :return: the mask, expanded to the necessary shape
        :rtype: :class:`numpy.ndarray`
        """
        new_mask_shape = [1 for x in shape]
        for i, s in enumerate(mask.shape):
            new_mask_shape[i] = s

        try:
            return np.broadcast_to(mask.reshape(new_mask_shape), shape)
        except ValueError:
            raise FilteringError('Could not expand mask with shape {} to shape of array ({}). If this array does not '
                                 'need filtered, directly index the SatelliteData instance. Otherwise, you will need '
                                 'to create a filter with a different broadcast_mask method.')

    def apply(self, data_array, dataset):
        """
        Apply this filter to a data array

        :param data_array: the array to be filtered
        :type data_array: :class:`numpy.ndarray`

        :param dataset: the handle to the dataset on disk, used to access other variables to create the mask

        :return: the data_array, either with poor quality values set to a fill value or as a masked array with those
         values masked
        :rtype: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`
        """
        mask = self.get_mask(dataset)
        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError('Expected boolean mask to be returned by get_mask(), got "{}" dtype instead'
                            .format(mask.dtype))
        mask = self.broadcast_mask(mask, data_array.shape)
        if self._use_masked_array:
            data_array = ma.masked_where(mask, data_array)
        else:
            data_array = data_array.copy()
            data_array[mask] = self._fill
        return data_array


class OCO2LiteFilter(SatelliteFilter):
    def __init__(self, masked=False, fill=np.nan, op_type=None):
        """
        Instantiate the filter.

        :param masked: controls whether the filter converts the data array into a masked array and masks bad quality
         data (``True``) or fills it with a fill value (``False``, default).
        :type masked: bool

        :param fill: fill value to use if ``masked`` is ``False``.

        :param op_type: what type of data to keep. This is a two-letter representation. The first letter can be 
         "L" for land, "O" for ocean, or "X" for any. The second letter can be "N" for nadir, "G" for glint, 
         "T" for target, or "X" for nadir, glint, or target. If not given, no filtering on operation/land-or-water
         will be done.
        """
        super(OCO2LiteFilter, self).__init__(masked=masked, fill=fill)
        self._op_type = op_type

    def get_mask(self, dataset):
        qual_mask = dataset['xco2_quality_flag'][:] > 0
        if self._op_type is None:
            return qual_mask

        op = self._op_type.upper()
        if op[0] == 'L':
            lw_indicator = (0,)
        elif op[0] == 'O':
            lw_indicator = (1,)
        elif op[0] == 'X':
            lw_indicator = (0,1,2,3)
        else:
            raise ValueError('"{}" is not a recognized first character for op_type'.format(op[0]))

        if op[1] == 'N':
            op_indicator = (0,)
        elif op[1] == 'G':
            op_indicator = (1,)
        elif op[1] == 'T':
            op_indicator = (2,)
        elif op[1] == 'X':
            # not including transition to/from target
            op_indicator = (0,1,2)

        lw_mask = ~np.isin(dataset['Sounding']['land_water_indicator'][:], lw_indicator)
        op_mask = ~np.isin(dataset['Sounding']['operation_mode'][:], op_indicator)
        return qual_mask | lw_mask | op_mask
