import unittest

import quadedge as qe


class TestQuadedge(unittest.TestCase):
    def test_quadedge(self):
        q = qe.QuadEdge()
        e = q.base

        self.assertIs(e, q.edges[0])
        self.assertIsNot(e, e.rot)
        self.assertIs(e._rot, e.rot)
        self.assertIs(e, e.rot.rot.rot.rot)
        self.assertIs(e.rot.rot.rot, e.inv_rot)

        self.assertIsNot(e, e.sym)
        self.assertIs(e, e.sym.sym)

        self.assertIs(e.o_next, e)
        self.assertIs(e.o_prev, e)
        self.assertIs(e.d_next, e)
        self.assertIs(e.d_prev, e)

        self.assertIs(e.l_next, e.sym)
        self.assertIs(e.l_prev, e.sym)
        self.assertIs(e.r_next, e.sym)
        self.assertIs(e.r_prev, e.sym)




if __name__ == '__main__':
    unittest.main()
