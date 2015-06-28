# this file taken from http://bugs.python.org/file9852/nonrecursivepickler-fixed.py
# for context see http://bugs.python.org/issue2480

__all__ = ['NonrecursivePickler']


import pickle

class LazySave( object ):
	'''Out of band marker for lazy saves among lazy writes.'''
	def __init__( self, obj ):
		self.obj = obj
	def __repr__( self ):
		return '<LazySave %s>'%repr( self.obj )

class LazyMemo( object ):
	'''Out of band marker for lazy memos among lazy writes.'''
	def __init__( self, obj ):
		self.obj = obj
	def __repr__( self ):
		return '<LazyMemo %s>'%repr( self.obj )

MEMOIMPORTANT = False	# turning this on creates pickles identical to the original implementation -- otherwise the memo ids are in a different order
class NonrecursivePickler( pickle.Pickler ):
	def __init__( self, file, protocol=None ):
		pickle.Pickler.__init__( self, file, protocol )
		self.lazywrites = []
		self.realwrite = file.write

		# Pickler.__init__ overwrites self.write, we do not want that
		del self.write
	def write( self, *args ):
		if self.lazywrites:
			self.lazywrites.append( args )
		else:
			self.realwrite( *args )
	def save( self, obj ):
		self.lazywrites.append( LazySave( obj ) )
	realsave = pickle.Pickler.save
	def lazymemoize( self, obj ):
		"""Store an object in the memo."""
		if self.lazywrites:
			self.lazywrites.append( LazyMemo( obj ) )
		else:
			self.realmemoize( obj )
	if MEMOIMPORTANT:
		memoize = lazymemoize
	realmemoize = pickle.Pickler.memoize
	def dump(self, obj):
		"""Write a pickled representation of obj to the open file."""
		if self.proto >= 2:
			self.write( pickle.PROTO + chr( self.proto ) )
		self.realsave( obj )
		while self.lazywrites:
			lws = self.lazywrites
			self.lazywrites = []
			while lws:
				lw = lws.pop( 0 )
				if type( lw ) is LazySave:
					self.realsave( lw.obj )
					if self.lazywrites:
						self.lazywrites.extend( lws )
						break
				elif type( lw ) is LazyMemo:
					self.realmemoize( lw.obj )
				else:
					self.realwrite( *lw )
		self.realwrite( pickle.STOP )

