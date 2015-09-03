import libpdalpython
import unittest

xml = open('../test/data/pipeline/pipeline_read.xml','r').read()
#print xml

class TestConstruction(unittest.TestCase):

  def test_construction(self):
      r = libpdalpython.PyPipeline(xml)

  def test_xml(self):
      r = libpdalpython.PyPipeline(xml)
      r.execute()
      self.assertEqual(len(r.xml), 2142)

  def test_arrays(self):
      r = libpdalpython.PyPipeline(xml)
      r.execute()
      arrays = r.arrays()
      a = arrays[0]
      print (a[0])
      print (a[1])
      import pdb;pdb.set_trace()
      self.assertEqual(len(arrays), 1)

def test_suite():
    return unittest.TestSuite(
        [TestConstruction])

if __name__ == '__main__':
    unittest.main()
