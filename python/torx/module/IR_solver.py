# Copyright 2019 The PytorX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch

class IrSolver(object):
    """
    This class solves IR drop in a crossbar array and calculates the output current w.r.t. wire resistence in the crossbar array.
    An example of using the solver is:
    vdd = 3.3
    Gsize = 64 # crxb size
    Gwire = 0.4 # wire conductance
    Gload = 10 # ADC and DAC loading conductance
    Gmin = 1/3e5
    Gmax = 1/3e2

    x = torch.rand(Gsize, 1, 1, 1, 1)*vdd # generating input
    Gmat = torch.rand(Gsize, Gsize, 1, 1)*(Gmax-Gmin)+Gmin # generating crxb
    iout_ideal = torch.matmul(Gmat.unsqueeze(4).permute(2, 3, 4, 1, 0), x.permute(2, 3, 4, 0 ,1)).cuda() # ideal current output
    crxb = IrSolver(Rsize=Gsize, Csize=Gsize, Gwire=Gwire, Gload=Gload, input_x=x, Gmat=Gmat)
    crxb.resetcoo()
    output_crxb = crxb.caliout()
    print(((iout_ideal - output_crxb)/iout_ideal*100).abs().max())# the max error%
    """

    def __init__(self, Rsize, Csize, Gwire, Gload, input_x, Gmat, device=torch.device("cuda:0")):
        """
        Initialize a crxb solver to calculate the iout change due to IR drop

        Args:
            Rsize (int): the row size of the crossbar
            Csize (int): the column size of the crossbar
            Gwire (float): the wire conductance of the crossbar
            input_x (float): the input voltage of the crossbar
            Gmat (Tensor.float): the weight matrix of the crossbar

        Returns:
            None
        """
        # input voltages
        self.input_x = input_x

        # crossbar
        self.Gmat = Gmat

        # ideal output current (no wire resistance)
        self.iout_ideal = torch.zeros(Csize, 1)

        self.mat_col = []
        self.mat_row = []
        self.mat_data = []

        self.GRsize = Rsize
        self.GCsize = Csize
        self.Gwire = Gwire
        self.Gload = Gload

        self.device = device.type

    def caliout(self):
        """
        This function is to calcuate the output of the current of the corssbar

        Args:
            None

        Retures:
            output current of the crossbar w.r.t. IR drop
        """
        if self.device == "cpu":
            current_mat = self._nodematgen()
        else:
            current_mat = self._nodematgen().cuda()

        # Generate the current array I of the MNA, which solve the node voltages using GV = I
        if self.device == "cpu":
            node_i = torch.LongTensor([self.mat_row, self.mat_col])
        else:
            node_i = torch.LongTensor([self.mat_row, self.mat_col]).cuda()
        node_v = torch.stack(self.mat_data)
        node_sp = torch.sparse.FloatTensor(node_i, node_v)

        nodes = torch.linalg.solve(node_sp.to_dense().permute(2, 3, 0, 1), current_mat.permute(2, 3, 0, 1, 4).contiguous().view(
                                   current_mat.size()[2], current_mat.size()[3], current_mat.size()[0], -1))
        
        # Solve batched linear systems
        temp = nodes.shape[2]
        outcurrent = nodes[:, :, temp - self.GCsize:temp, :]
        del nodes
        try:
            outcurrent = outcurrent * self.Gload
        except:
            outcurrent = outcurrent * self.Gload
        return outcurrent

    def resetcoo(self):
        """
        This function resets the coo matrix for a new calculation.

        Args:
            None

        Returns:
            None
        """
        self.mat_col = []
        self.mat_row = []
        self.mat_data = []

    def _add_data(self, row_data, col_data, data_data):
        """
        This function adds elements to the coo matrix

        Args:
            row_data (int): the row coordinate of the coo matrix
            col_data (int): the column coordinate of the coo matrix
            data_data (float): the entries of the w.r.t. the row and column coordinate

        Returns:
            None
        """
        self.mat_row.append(row_data)
        self.mat_col.append(col_data)
        if self.device == "cpu":
            self.mat_data.append(data_data)
        else:
            self.mat_data.append(data_data.cuda())

    def _nodematgen(self):
        """
        This function generates the node conductance matrix. The node conductance matrix is batched
        according to dimension of the input tensors. The detailed descrapition of the node conductance matrix please to
        this link: https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA1.html

        Args:
            None

        Returns:
            The conductance matrix in coo format.
            current_mat (tensor.float): the current matrix.
        """
        current_mat = torch.zeros(self.GRsize ** 2 * 2, self.input_x.shape[1], self.input_x.shape[2], self.input_x.shape[3], self.input_x.shape[4])
        extender = torch.ones(self.Gmat.size()[2], self.Gmat.size()[3])

        electrode = ['top', 'bot']
        counter = 0

        for row in range(self.GRsize):
            for ele in electrode:
                for col in range(self.GCsize):
                    if col == 0 and ele == 'top':  # edge type I
                        current_mat[counter] = self.input_x[row] * self.Gload
                        self._add_data(counter, counter, self.Gload + self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter + 1, -self.Gwire * extender)
                        self._add_data(counter, counter + self.GRsize, -self.Gmat[row][col])

                    elif row == 0 and ele == 'bot':  # edge type II
                        self._add_data(counter, counter, self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter + 2 * self.GRsize, -self.Gwire * extender)
                        self._add_data(counter, counter - self.GRsize, -self.Gmat[row][col])

                    elif col == self.GCsize - 1 and ele == 'top':  # edge type III
                        self._add_data(counter, counter, self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter - 1, -self.Gwire * extender)
                        self._add_data(counter, counter + self.GRsize, -self.Gmat[row][col])

                    elif row == self.GRsize - 1 and ele == 'bot':  # edge type IV
                        self._add_data(counter, counter, self.Gload + self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter - 2 * self.GRsize, -self.Gwire * extender)
                        self._add_data(counter, counter - self.GRsize, -self.Gmat[row][col])

                    else:
                        if ele == 'top':
                            self._add_data(counter, counter, self.Gmat[row][col] + 2 * self.Gwire)
                            self._add_data(counter, counter + 1, -self.Gwire * extender)
                            self._add_data(counter, counter - 1, -self.Gwire * extender)
                            self._add_data(counter, counter + self.GCsize, -self.Gmat[row][col])

                        elif ele == 'bot':
                            self._add_data(counter, counter, self.Gmat[row][col] + 2 * self.Gwire)
                            self._add_data(counter, counter + (2 * self.GRsize), -self.Gwire * extender)
                            self._add_data(counter, counter - (2 * self.GRsize), -self.Gwire * extender)
                            self._add_data(counter, counter - self.GRsize, -self.Gmat[row][col])

                    counter += 1

        return current_mat
