#coding: utf-8

import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

#Define function for convolutional and pool layer
class LeNetConvPoolLayer( object ):

    def __init__( self, rng, input, image_shape, filter_shape, poolsize = ( 2, 2 ) ):

        #input feature map number == filter number
        assert image_shape[ 1 ] == filter_shape[ 1 ]

        fan_in = np.prod( filter_shape[ 1 : ] )
        fan_out = filter_shape[ 0 ] * np.prod( filter_shape[ 2 : ] ) / np.prod( poolsize )
        
        W_bound = np.sqrt( 6.0 / ( fan_in + fan_out ) )
        #node_num = np.prod( image_shape[ 1 : ] )
        #W_bound = np.sqrt( 1.0 / node_num )

        self.W = theano.shared( 

            np.asarray( rng.uniform( low = - W_bound, high = W_bound, size = filter_shape ), 
                        
                        dtype = theano.config.floatX ),
            
            borrow = True )

        b_values = np.zeros( ( filter_shape[ 0 ], ), dtype = theano.config.floatX )
        self.b = theano.shared( value = b_values, borrow = T )

        conv_out = conv.conv2d(

            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            image_shape = image_shape )

        pooled_out = downsample.max_pool_2d(

            input = conv_out,
            ds = poolsize,
            ignore_border = True )

        self.output = T.tanh( pooled_out + self.b.dimshuffle( 'x', 0, 'x', 'x' ) )
        #self.output = pooled_out + self.b.dimshuffle( 'x', 0, 'x', 'x' ) 
        self.params = [ self.W, self.b ]

#Define function for convolutional and pool layer
class VGGConvPoolLayer( object ):

    def __init__( self, rng, input, image_shape, filter_shape, poolsize = ( 2, 2 ) ):
        
        #input feature map number == filter number
        assert image_shape[ 1 ] == filter_shape[ 1 ]

        fan_in = np.prod( filter_shape[ 1 : ] )
        fan_out = filter_shape[ 0 ] * np.prod( filter_shape[ 2 : ] ) / np.prod( poolsize )
        
        W_bound = np.sqrt( 6.0 / ( fan_in + fan_out ) )
        self.W = theano.shared( 

            np.asarray( rng.uniform( low = - W_bound, high = W_bound, size = filter_shape ), 
                        
                        dtype = theano.config.floatX ),
            
            borrow = True, name = 'W' )

        b_values = np.zeros( ( filter_shape[ 0 ], ), dtype = theano.config.floatX )
        self.b = theano.shared( value = b_values, borrow = T )

        conv_out = conv.conv2d(

            input = input,
            filters = self.W )

        pooled_out = downsample.max_pool_2d(

            input = conv_out,
            ds = poolsize,
            ignore_border = True )

        self.output = T.tanh( pooled_out + self.b.dimshuffle( 'x', 0, 'x', 'x' ) )
        #self.output = pooled_out + self.b.dimshuffle( 'x', 0, 'x', 'x' ) 
        self.params = [ self.W, self.b ]
        #self.params = [ self.b ]

