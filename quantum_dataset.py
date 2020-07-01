from abc import ABC, abstractmethod
import os
import random
import h5py
import pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from scipy import spatial as sp
from scipy.io import loadmat

from rdkit import Chem

from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from torch import as_tensor, cat


class Molecule(ABC):
    """A class for creating a rdmol obj and coulomb matrix from a smile.
    Subclass and implement load_data()"""
    
    atomic_n = {'C': 6, 'H': 1, 'N': 7, 'O': 8, 'F': 9}
    
    def __init__(self, in_dir):
        self.load_data(in_dir)
        self.rdmol_from_smile(self.smile)
        self.create_adjacency(self.rdmol)
        self.create_distance(self.xyz)
        self.create_coulomb(self.distance, self.xyz)
        
    @abstractmethod
    def __repr__(self):
        return self.mol_id
        
    @abstractmethod
    def load_data(self):
        self.smile = '' 
        self.n_atoms = 0  
        self.properties = []  
        self.xyz = []  # [['atom_type',x,y,z],...]
        
    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data
        
    def rdmol_from_smile(self, smile):
        self.rdmol = Chem.AddHs(Chem.MolFromSmiles(smile))
    
    def create_adjacency(self, rdmol):
        """use the rdmol mol block adjacency list to create a nxn symetric matrix with 0, 1, 2 or
        3 for bond type where n is the indexed atom list for the molecule"""
        block = Chem.MolToMolBlock(rdmol)
        self.adjacency = np.zeros((self.n_atoms, self.n_atoms), dtype='float32')
        block = block.strip(' ').split('\n')
        for b in block:
            b = b.split()
            if len(b) == 4:
                self.adjacency[(int(b[0])-1),(int(b[1])-1)] = int(b[2]) # shift -1 to index from zero
                self.adjacency[(int(b[1])-1),(int(b[0])-1)] = int(b[2]) # create bi-directional connection
             
    def create_distance(self, xyz):
        m = np.zeros((len(xyz), 3))
        for i, atom in enumerate(xyz):
            m[i,:] = [float(np.char.replace(x, '*^', 'e')) for x in atom[1:4]] # fix the scientific notation
        self.distance = sp.distance.squareform(sp.distance.pdist(m)).astype('float32')
      
    def create_coulomb(self, distance, xyz, sigma=1):
        """creates coulomb matrix obj attr.  set sigma to False to turn off random sorting.  
        sigma = stddev of gaussian noise.
        https://papers.nips.cc/paper/4830-learning-invariant-representations-of-\
        molecules-for-atomization-energy-prediction"""
        atoms = []
        for atom in xyz:
            atoms.append(Molecule.atomic_n[atom[0]]) 
        atoms = np.asarray(atoms, dtype='float32')
        qmat = atoms[None, :]*atoms[:, None]
        idmat = np.linalg.inv(distance)
        np.fill_diagonal(idmat, 0)
        coulomb = qmat@idmat
        np.fill_diagonal(coulomb, 0.5 * atoms ** 2.4)
        if sigma:  
            self.coulomb = self.sort_permute(coulomb, sigma)
        else:  
            self.coulomb = coulomb
    
    def sort_permute(self, matrix, sigma):
        norm = np.linalg.norm(matrix, axis=1)
        noised = np.random.normal(norm, sigma)
        indexlist = np.argsort(noised)
        indexlist = indexlist[::-1]  # invert
        return matrix[indexlist][:,indexlist]

    
class QM9Mol(Molecule):
    
    def __repr__(self):
        return self.in_file[:-4]
    
    def load_data(self, in_file):
        """load from the .xyz files of the qm9 dataset
        (http://quantum-machine.org/datasets/)
        properties = ['A','B','C','mu','alpha','homo','lumo', 
                      'gap','r2','zpve','U0','U','H','G','Cv']
        """
        self.in_file = in_file
        xyz = self.open_file(in_file)
        self.smile = xyz[-2]
        self.n_atoms = int(xyz[0])
        self.properties = xyz[1].strip().split('\t')[1:] # [float,...]
        
        self.xyz = []
        for atom in xyz[2:self.n_atoms+2]:
            self.xyz.append(atom.strip().split('\t')) # [['atom_type',x,y,z,mulliken],...]
            
        self.mulliken = []
        for atom in self.xyz:
            m = np.reshape(np.asarray(np.char.replace(atom[4], '*^', 'e'), 
                                                          dtype=np.float32), -1)
            self.mulliken.append(m)
        self.mulliken = np.concatenate(self.mulliken, axis=0)
        
class QDataset(Dataset, ABC):
    """An abstract base class for quantum datasets"""
    @abstractmethod
    def __init__(self, in_dir):
        self.load_data(in_dir)
        self.embeddings = []  # [(n_vocab, len_vec, param.requires_grad),...]
        self.ds_idx = []  # list of the dataset's indices
    
    @abstractmethod
    def __getitem__(self, i):  # set X and y and do preprocessing here
        return x_con[i], x_cat[i], target[i]  # continuous, categorical, target.  empty list if none.
    
    @abstractmethod
    def __len__(self):
        return len(self.ds_idx)
    
    @abstractmethod
    def load_data(self):
        return data
        
    
class QM7(QDataset):
    """http://quantum-machine.org/datasets/
    This dataset is a subset of GDB-13 (a database of nearly 1 billion stable 
    and synthetically accessible organic molecules) composed of all molecules of 
    up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. 
    We provide the Coulomb matrix representation of these molecules and their atomization 
    energies computed similarly to the FHI-AIMS implementation of the Perdew-Burke-Ernzerhof 
    hybrid functional (PBE0). This dataset features a large variety of molecular structures 
    such as double and triple bonds, cycles, carboxy, cyanide, amide, alcohol and epoxy.
    
    https://arxiv.org/abs/1904.10321
    Prediction of the Atomization Energy of Molecules Using Coulomb Matrix and Atomic 
    Composition in a Bayesian Regularized Neural Networks
    """
    def __init__(self, in_file = './data/qm7/qm7.mat'):
        self.load_data(in_file)
        self.embeddings = []
        self.x_cat = []
        
    def __getitem__(self, i): 
        return as_tensor(np.reshape(self.coulomb[i,:,:], -1)), self.x_cat, \
                    as_tensor(np.reshape(self.ae[:,i], -1))
      
    def __len__(self):
        return len(self.ds_idx)
    
    def load_data(self, in_file):
        qm7 = loadmat(in_file)
        self.coulomb = qm7['X'] # (7165, 23, 23)
        self.xyz = qm7['R'] # (7165, 3)
        self.atoms = qm7['Z'] # (7165, 23)
        self.ae = qm7['T'] # (1, 7165) atomization energy
        self.ds_idx = list(range(1, self.coulomb.shape[0]))
        
        
class QM7b(QDataset):
    """http://quantum-machine.org/datasets/
    This dataset is an extension of the QM7 dataset for multitask learning where 13 
    additional properties (e.g. polarizability, HOMO and LUMO eigenvalues, excitation 
    energies) have to be predicted at different levels of theory (ZINDO, SCS, PBE0, GW). 
    Additional molecules comprising chlorine atoms are also included, totalling 7211 molecules.
    
    properties: atomization energies, static polarizabilities (trace of tensor) α, frontier 
    orbital eigenvalues HOMO and LUMO, ionization potential, electron affinity, optical 
    spectrum simulations (10nm-700nm) first excitation energy, optimal absorption maximum, 
    intensity maximum.
    
    https://th.fhi-berlin.mpg.de/site/uploads/Publications/QM-NJP_20130315.pdf
    Machine Learning of Molecular Electronic Properties in Chemical Compound Space
    """
    properties = ['E','alpha_p','alpha_s','HOMO_g','HOMO_p','HOMO_z',
                  'LUMO_g','LUMO_p','LUMO_z','IP','EA','E1','Emax','Imax']
   
    def __init__(self, target, features=[], in_file='./data/qm7/qm7b.mat'):
        self.features = features
        self.target = target
        self.embeddings = []
        self.x_cat = []
        self.load_data(target, features, in_file)
        
    def __getitem__(self, i): 
        flat_c = np.reshape(self.coulomb[i-1,:,:], -1).astype(np.float32)
        x_con = np.concatenate((flat_c, 
                    self.properties[self.features].iloc[i].astype(np.float32)), axis=0)
        return as_tensor(x_con), self.x_cat, as_tensor(self.y[:,i-1])
      
    def __len__(self):
        return len(self.ds_idx)  
    
    def load_data(self, target, features, in_file):
        qm7b = loadmat(in_file)
        self.coulomb = qm7b['X'] # (7211, 23, 23)
        self.properties = pd.DataFrame(data=qm7b['T'], dtype=np.float32, 
                                       columns=QM7b.properties) # (7211, 14)
        self.y = self.properties.pop(self.target).values.reshape(1, -1) # (1, 7211) 
        self.ds_idx = list(range(self.coulomb.shape[0]))
         
        
class QM9(QDataset):
    """http://quantum-machine.org/datasets/
   
    dsgdb9nsd.xyz.tar.bz2    - 133885 molecules with properties in XYZ-like format
    dsC7O2H10nsd.xyz.tar.bz2 - 6095 isomers of C7O2H10 with properties in XYZ-like format
    validation.txt           - 100 randomly drawn molecules from the 133885 set with enthalpies of formation
    uncharacterized.txt      - 3054 molecules from the 133885 set that failed a consistency check
    atomref.txt              - Atomic reference data
    readme.txt               - Documentation

    1          Number of atoms na
    2          Properties 1-17 (see below)
    3,...,na+2 Element type, coordinate (x,y,z) (Angstrom), and Mulliken partial charge (e) of atom
    na+3       Frequencies (3na-5 or 3na-6)
    na+4       SMILES from GDB9 and for relaxed geometry
    na+5       InChI for GDB9 and for relaxed geometry

    The properties stored in the second line of each file:

    I.  Property  Unit         Description
    --  --------  -----------  --------------
     1  tag       -            "gdb9"; string constant to ease extraction via grep
     2  index     -            Consecutive, 1-based integer identifier of molecule
     3  A         GHz          Rotational constant A
     4  B         GHz          Rotational constant B
     5  C         GHz          Rotational constant C
     6  mu        Debye        Dipole moment
     7  alpha     Bohr^3       Isotropic polarizability
     8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
     9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
    10  gap       Hartree      Gap, difference between LUMO and HOMO
    11  r2        Bohr^2       Electronic spatial extent
    12  zpve      Hartree      Zero point vibrational energy
    13  U0        Hartree      Internal energy at 0 K
    14  U         Hartree      Internal energy at 298.15 K
    15  H         Hartree      Enthalpy at 298.15 K
    16  G         Hartree      Free energy at 298.15 K
    17  Cv        cal/(mol K)  Heat capacity at 298.15 K
    
    https://www.nature.com/articles/sdata201422
    Quantum chemistry structures and properties of 134 kilo molecules
    
    https://arxiv.org/abs/1809.02723
    Deep Neural Network Computes Electron Densities and Energies of a Large Set of 
    Organic Molecules Faster than Density Functional Theory (DFT)
    
    https://arxiv.org/abs/1908.00971
    Physical machine learning outperforms "human learning" in Quantum Chemistry
    
    """
    LOW_CONVERGENCE = [21725,87037,59827,117523,128113,129053,129152, 
                       129158,130535,6620,59818]
    
    properties = ['A','B','C','mu','alpha','homo','lumo', 
                  'gap','r2','zpve','U0','U','H','G','Cv']
    
    def __init__(self, in_dir='./data/qm9/qm9.xyz/', n=133885, 
                 features=[], target='', dim=29, use_pickle=True):
        """dim = length of longest molecule that all molecules will be padded to
        features/target = QM9.properties, 'coulomb', 'mulliken', QM9Mol.attr
        """
        self.features, self.target, self.dim = features, target, dim
        self.datadic = self.load_data(in_dir, n, use_pickle)
        # filter here
        self.ds_idx = list(self.datadic.keys())
        self.embeddings = []
        self.x_cat = [] # no categorical features
    
    def __getitem__(self, i):
        x_con, x_cat, y = self.load_mol(i)
        return as_tensor(np.reshape(x_con, -1)), x_cat, \
                    as_tensor(np.reshape(y, -1))
        
    def __len__(self):
        return len(self.ds_idx)
       
    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data
        
    def load_data(self, in_dir, n, use_pickle): # n = non random subset selection (for testing)
        
        if os.path.exists('./data/qm9/qm9_datadic.p') and use_pickle:
            print('loading QM9 datadic from a pickled copy...')
            datadic = pickle.load(open('./data/qm9/qm9_datadic.p', 'rb'))
        else:
            datadic = {}
            for filename in sorted(os.listdir(in_dir)):
                if filename.endswith('.xyz'):
                    datadic[int(filename[-10:-4])] = QM9Mol(in_dir+filename)
                    if len(datadic) % 10000 == 1: print('QM9 molecules created:', len(datadic))
                    if len(datadic) > n - 1:
                        break
                       
            unchar = self.get_uncharacterized()
            for mol in unchar: 
                try: del datadic[mol]
                except: continue
            print('total QM9 molecules created:', len(datadic))
            
            if use_pickle:
                print('pickling a copy of the QM9 datadic...')        
                pickle.dump(datadic, open('./data/qm9/qm9_datadic.p', 'wb'))
                
        return datadic
    
    def get_uncharacterized(self, in_file='./data/qm9/uncharacterized.txt'):
        """uncharacterized.txt - 3054 molecules from the 133885 set that failed a 
        consistency check.  Returns a list of ints of the 3054 molecules (datadic keys)"""
        data = self.open_file(in_file)
        unchar = []
        for mol in data[8:]:
            for m in mol.strip().split():
                if m.isdigit():
                    unchar.append(int(m))
        return unchar
    
    def load_mol(self, idx):
        mol = self.datadic[idx]
        
        def load_feature(feature):
            if fea == 'coulomb': 
                flat = np.reshape(mol.coulomb, -1)
                return np.pad(flat, (0, self.dim**2-len(mol.coulomb)**2))
            elif fea == 'mulliken':
                return np.pad(mol.mulliken, (0, self.dim-len(mol.mulliken)))
            elif fea in QM9.properties: 
                return np.reshape(np.asarray(mol.properties[QM9.properties.index(fea)],
                                                                   dtype=np.float32), -1)
            else: 
                return np.reshape(np.asarray(getattr(mol, fea), dtype=np.float32), -1)
                
        feats = []
        for fea in self.features:
            feats.append(load_feature(fea))
       
        x_con = np.concatenate(feats, axis=0)
        y = load_feature(self.target)
        
        return x_con, self.x_cat, y
            
        
class Champs(QDataset):
    """https://www.kaggle.com/c/champs-scalar-coupling
    85003 molecules, 1533536 atoms, 4658146 couplings, 2505542 test couplings
    
    potential_energy.csv ['molecule_name','potential_energy'] 
    scalar_coupling_contributions.csv ['molecule_name','atom_index_0','atom_index_1','type','fc','sd','pso','dso'] 
    train.csv ['id','molecule_name','atom_index_0','atom_index_1','type','scalar_coupling_constant'] 
    dipole_moments.csv ['molecule_name','X','Y','Z'] 
    mulliken_charges.csv ['molecule_name','atom_index','mulliken_charge'] 
    sample_submission.csv ['id','scalar_coupling_constant'] 
    structures.csv ['molecule_name','atom_index','atom','x','y','z'] 
    test.csv ['id', 'molecule_name','atom_index_0','atom_index_1','type'] n=2505542

    TODO atom_idx vs coulomb idx significance
    TODO make forward as well as reverse connections selected for test set (use id)
    """
    files = ['magnetic_shielding_tensors.csv', 'potential_energy.csv', 
             'scalar_coupling_contributions.csv', 'train.csv', 'dipole_moments.csv', 
             'mulliken_charges.csv', 'sample_submission.csv', 'structures.csv', 'test.csv']
    types = ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']
    atomic_n = {'C': 6, 'H': 1, 'N': 7, 'O': 8, 'F': 9}
    
    def __init__(self, in_dir='./data/champs/', n=4658147, features=True, use_h5=False, infer=False):
        self.in_dir = in_dir
        self.embeddings = [(8,128,True),(32,32,False),(4,64,True),(32,32,False),(4,64,True)]  
        self.con_ds, self.cat_ds, self.target_ds = self.load_data(self.in_dir, features, use_h5, infer)
        self.ds_idx = list(range(len(self.target_ds)))
        
    def __getitem__(self, i):
        
        def to_torch(ds, i):
            if len(ds) == 0:
                return []
            else: return as_tensor(ds[i])
           
        x_con = to_torch(self.con_ds, i)
        x_cat = to_torch(self.cat_ds, i)
        y = to_torch(self.target_ds, i)
        return x_con, x_cat, y
    
    def __len__(self):
        return len(self.ds_idx)
    
    def load_data(self, in_dir, features, use_h5, infer):

        if infer:
            df = pd.read_csv(in_dir+'test.csv', header=0, names=['id','molecule_name', 
                   'atom_index_0','atom_index_1','type'], index_col=False)
            target_ds = df['id'].values.astype('int64')
            
        else:
            df = pd.read_csv(in_dir+'train.csv', header=0, names=['id','molecule_name', 
                 'atom_index_0','atom_index_1','type','scalar_coupling_constant'], index_col=False)
            target_ds = df.pop('scalar_coupling_constant').astype('float32')
            
#             pe = pd.read_csv(in_dir+'potential_energy.csv', header=0, names=['molecule_name',
#                                                  'potential_energy'], index_col=False)
#             mulliken = pd.read_csv(in_dir+'mulliken_charges.csv', header=0, names=['molecule_name',
#                                'atom_index','mulliken_charge'], index_col=False)
            
        structures = pd.read_csv(in_dir+'structures.csv', header=0, names=['molecule_name',
                             'atom_index','atom','x','y','z'], index_col=False)
        df = df.merge(structures, how='left', left_on=['molecule_name','atom_index_0'],
                                              right_on=['molecule_name','atom_index'],
                                              suffixes=('_0','_1'))
        df = df.merge(structures, how='left', left_on=['molecule_name','atom_index_1'],
                                              right_on=['molecule_name','atom_index'],
                                              suffixes=('_0','_1'))

        df.columns = ['id', 'molecule_name','atom_index_0_drop','atom_index_1_drop','type',
                      'atom_index_0','atom_0','x_0','y_0','z_0','atom_index_1','atom_1',
                      'x_1','y_1','z_1']

        df = df.drop(columns=['atom_index_0_drop','atom_index_1_drop'])

        df = df[['id','molecule_name','type','atom_index_0','atom_0','x_0','y_0','z_0',
                 'atom_index_1','atom_1','x_1','y_1','z_1']]

        if not infer:
            print('bing: if not infer')
            df = pd.concat([df, target_ds], axis=1)        
            # create reverse connections           
            rev = df.copy()
            rev.columns = ['id', 'molecule_name','type','atom_index_1','atom_1',
                           'x_1','y_1','z_1','atom_index_0','atom_0','x_0','y_0',
                           'z_0','scalar_coupling_constant']
            rev = rev[['id','molecule_name','type', 'atom_index_0','atom_0','x_0',
                       'y_0','z_0','atom_index_1','atom_1','x_1','y_1','z_1',
                       'scalar_coupling_constant']]
            df = pd.concat([df, rev])
            target_ds = df.pop('scalar_coupling_constant').values.astype('float32')
           
        categorical = ['type','atom_index_0','atom_0','atom_index_1','atom_1']
        continuous = ['x_0','y_0','z_0','x_1','y_1','z_1']
        if not features:
            continuous = []
        
        df[categorical] = df[categorical].astype('category')
        df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
        df[categorical] = df[categorical].astype('int64')
        df[continuous] = df[continuous].astype('float32')

        con_ds = df[continuous].values
        cat_ds = df[categorical].values
          
        lookup = df.pop('molecule_name').str.slice(start=-6).astype('int64')
       
        if use_h5:
            print('creating Champs h5 dataset...')
            with h5py.File(in_dir+'champs_cat.h5', 'w') as h5p:
                cat_ds = h5p.create_dataset('x_cat', data=cat_ds, chunks=True)[()]  # index in with empty tuple [()]
            with h5py.File(in_dir+'champs_con.h5', 'w') as h5p:
                con_ds = h5p.create_dataset('x_con', data=con_ds, chunks=True)[()]
            with h5py.File(in_dir+'champs_target.h5', 'w') as h5p:
                target_ds = h5p.create_dataset('target', data=target_ds, chunks=True)[()]
            with h5py.File(in_dir+'champs_lookup.h5', 'w') as h5p:
                self.lookup = h5p.create_dataset('lookup', data=lookup, chunks=True)[()]
        else: 
            self.lookup = lookup

        return con_ds, cat_ds, np.reshape(target_ds, (-1, 1))

    @classmethod
    def inspect_csv(cls, in_dir='./data/'): 
        feature_labels = {}
        for f in Champs.files:
            out = pd.read_csv(in_dir + f)
            print(f, '\n')
            print(out.info(), '\n')
            print(out.head(5), '\n')
            print(out.describe(), '\n')
            feature_labels[f] = list(out.columns)
            del out
            
        for fea in feature_labels:
            print(fea, feature_labels[fea], '\n')

class SuperSet(QDataset):
    
    def __init__(self, PrimaryDS, SecondaryDS, p_params, s_params):
        self.pds = PrimaryDS(**p_params)
        self.sds = SecondaryDS(**s_params)
        
        self.embeddings = self.pds.embeddings + self.sds.embeddings
        self.ds_idx = self.pds.ds_idx 
        
    def __getitem__(self, i):
        # lookup the molecule name used by the primary ds and use it to select data from 
        # the secondary ds and then concatenate both outputs and return it
        x_con1, x_cat1, y1 = self.pds[i]
        x_con2, x_cat2, y2 = self.sds[self.pds.lookup.iloc[i]]  # TODO H5 ds uses numpy indexing
       
        def concat(in1, in2, dim=0):
            try:
                return cat([in1, in2], dim=dim)
            except:
                if len(in1) != 0: return in1
                elif len(in2) != 0: return in2
                else: return []
                
        x_con = concat(x_con1, x_con2)
        x_cat = concat(x_cat1, x_cat2)
        return x_con, x_cat, y1
        
    def __len__(self):
        return len(self.ds_idx)
    
    def load_data(self):
        pass
        
    