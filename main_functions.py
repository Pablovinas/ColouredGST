from math import exp, sin, cos, sqrt, pi
import cmath
from qiskit.quantum_info import Pauli
import numpy as np
import pygsti
from scipy.linalg import expm

# Pauli matrices
pauli_i, pauli_x, pauli_y, pauli_z = Pauli('I').to_matrix(), Pauli('X').to_matrix(), Pauli('Y').to_matrix(), Pauli('Z').to_matrix()
pauli_vector = np.array([pauli_i, pauli_x, pauli_y, pauli_z])

def ptm_state(coefs):

    """
    Calculates the density matrix given a set of coefficients.

    Parameters:
    coefs (array-like): The coefficients used to calculate the density matrix.

    Returns:
    rho (ndarray): The calculated state in PTM representation.

    If the length of the coefficients is not 4, a default coefficient of 1 is inserted at the beginning.
    The density matrix is calculated using the coefficients and the pauli_vector.

    """

    if len(coefs) != 4:
        coefs = np.insert(coefs, 0, 1)

    state = 1/sqrt(2)*np.array(coefs)

    return state 

def density_matrix(coefs):

    """
    Calculates the density matrix given a set of coefficients.

    Parameters:
    coefs (array-like): The coefficients used to calculate the density matrix.

    Returns:
    rho (ndarray): The calculated density matrix.

    If the length of the coefficients is not 4, a default coefficient of 1 is inserted at the beginning.
    The density matrix is calculated using the coefficients and the pauli_vector.

    """

    # Insert a default coefficient of 1 if the length is not 4
    if len(coefs) != 4:
        coefs = np.insert(coefs, 0, 1)

    # Calculate the density matrix using the coefficients and the pauli_vector
    rho = 1/2 * np.einsum('i,ikl->kl', coefs, pauli_vector)

    return rho

def quantum_evolution(initial_state, matrix, basis = pauli_vector, representation = 'PTM'):

    """
    Perform quantum evolution on the initial state using the given matrix and representation.

    Parameters:
        initial_state (numpy.ndarray): The initial state vector.
        matrix (numpy.ndarray): The matrix representing the quantum evolution operator.
        basis (numpy.ndarray, optional): The basis vectors. Defaults to Pauli basis.
        representation (str, optional): The representation to use, Process Matrix (chi) or Pauli
        Transfer Matrix (PTM). Defaults to 'PTM'.

    Returns:
        numpy.ndarray: The final state vector after quantum evolution.
    """

    if representation == 'PTM':
        # Perform quantum evolution using PTM representation
        final_state = matrix @ initial_state

    elif representation == 'chi':
        # Perform quantum evolution using chi matrix representation
        basis_conj_T = np.conj(basis).swapaxes(-2, -1)
        temp = np.einsum('ijk,kl->ijl', basis, initial_state)
        temp = np.einsum('ijk,mkl->imjl', temp, basis_conj_T)
        final_state = np.einsum('ij,ijkl->kl', matrix, temp)

    return final_state


def chi2ptm(chi, basis = pauli_vector):

    """
    Calculates the Pauli Transfer Matrix (PTM) from a given Process Matrix.

    Parameters:
    chi (numpy.ndarray): The chi matrix.
    basis (numpy.ndarray, optional): The basis to be used for the quantum evolutions. Defaults to Pauli basis.

    Returns:
    numpy.ndarray: The Pauli Transfer Matrix (PTM).
    """

    # Perform quantum evolutions for each element in the basis
    quantum_evolutions = [quantum_evolution(element, chi, basis, representation='chi') for element in basis]

    # Convert the list of quantum evolutions into a numpy array
    quantum_evolutions = np.array(quantum_evolutions)

    # Perform einsum operations to calculate the PTM
    temp = np.einsum('ijk,mkl->imjl', basis, quantum_evolutions)
    PTM = 0.5 * np.einsum('imll->im', temp)
    
    return PTM

import numpy as np

def ptm2chi(PTM, basis = pauli_vector):

    """
    Converts a Pauli Transfer Matrix (PTM) to a Chi matrix using linear Quantum Process Tomography (see Nielsen & 
    Chuang page 389).

    Parameters:
    PTM (numpy.ndarray): The Pauli Transfer Matrix.
    basis (numpy.ndarray, optional): The basis to be used for the quantum evolutions. Defaults to Pauli basis.

    Returns:
    numpy.ndarray: The Process matrix.
    """

    # Define the initial states
    initial_states = np.array([[1, 0, 0, 1], [0, 1, 1j, 0], [0, 1, -1j, 0], [1, 0, 0, -1]])

    # Evolve the initial states using the PTM
    evolved_states = [quantum_evolution(initial_state, PTM, basis, representation='PTM') for initial_state in initial_states]

    # Create the density matrix of the evolved states
    evolved_states_density = np.block([
        [density_matrix(evolved_states[0]), density_matrix(evolved_states[1])],
        [density_matrix(evolved_states[2]), density_matrix(evolved_states[3])]
    ])

    # Define the Lambda matrix
    Lambda = 0.5 * np.block([[pauli_i, pauli_x],[pauli_x, -pauli_i]])

    # Calculate the Chi matrix
    chi = Lambda @ evolved_states_density @ Lambda

    # Basis change -i\sigma_y -> \sigma_y
    chi[:,2] = 1j*chi[:,2]
    chi[2,:] = - 1j*chi[2,:]

    return chi

def build_gate(params, pulse_duration, p, markovian_approx = True, representation = 'PTM', phase = 0):

    """
    Builds a gate using the given parameters with the coloured phase noise parameterization.

    Args:
        params (np.array): An array of parameters Gamma_1, Delta_1 (Gamma_2, Delta_2).
        pulse_duration (float): The duration of the pulse.
        p (float):Number of times the single gate is repeated in the long-sequence scheme (unused).
        markovian_approx (bool, optional): Whether to use the Markovian approximation. Defaults to True.
        representation (str, optional): The representation to use. Defaults to 'PTM'.
        phase (float, optional): The phase value. Defaults to 0.

    Returns:
        numpy.ndarray: The gate matrix.

    Raises:
        ValueError: If an invalid phase value is provided.
    """

    if markovian_approx:
        params[2:] = 0

    Gamma_1, Delta_1, Gamma_2, Delta_2 = p * params[0], p * params[1], p * params[2], p * params[3]

    #theta
    theta = 1 / 2 * cmath.sqrt(Delta_1 ** 2 - Delta_2 ** 2 - Gamma_2 ** 2)

    # Calculate chi_A
    chi_A = (1 / 4) * ((1 + cmath.exp(-Gamma_1)) * pauli_i + \
        2 * cmath.exp(-Gamma_1 / 2) * (cmath.cos(p * pulse_duration) * cmath.cos(theta) + (Delta_1 / (2 * theta)) * cmath.sin(p * pulse_duration) * cmath.sin(theta)) * pauli_z - \
        2 * cmath.exp(-Gamma_1 / 2) * (cmath.sin(p * pulse_duration) * cmath.cos(theta) + (Delta_1 / (2 * theta)) * cmath.cos(p * pulse_duration) * cmath.sin(theta)) * pauli_y)

    # Calculate chi_B
    chi_B = (1 / 4) * ((1 - cmath.exp(-Gamma_1)) * pauli_i - \
        cmath.exp(-Gamma_1 / 2) * (cmath.sin(theta) / theta) * (Gamma_2 * cmath.cos(p * pulse_duration) + Delta_2 * cmath.sin(p * pulse_duration)) * pauli_z - \
        cmath.exp(-Gamma_1 / 2) * (cmath.sin(theta) / theta) * (Gamma_2 * cmath.sin(p * pulse_duration) - Delta_2 * cmath.cos(p * pulse_duration)) * pauli_x)

    # Create the block matrix
    chi = np.zeros((4, 4), dtype=complex)
    chi[:2, :2] = chi_A
    chi[2:, 2:] = chi_B

    if phase == 0:
        pass

    elif phase == pi / 2:
        chi[1, 1], chi[2, 2] = chi[2, 2], chi[1, 1]
        chi[0, 2], chi[1, 3] = -chi[0, 1], chi[2, 3]
        chi[2, 0], chi[3, 1] = -chi[0, 2], chi[1, 3]
        chi[0, 1], chi[2, 3], chi[1, 0], chi[3, 2] = 0, 0, 0, 0

    elif phase == pi:
        chi[0, 1], chi[1, 0], chi[2, 3], chi[3, 2] = -chi[0, 1], -chi[1, 0], -chi[2, 3], -chi[3, 2]

    elif phase == 3 * pi / 2:
        chi[1, 1], chi[2, 2] = chi[2, 2], chi[1, 1]
        chi[0, 2], chi[1, 3] = chi[0, 1], -chi[2, 3]
        chi[2, 0], chi[3, 1] = -chi[0, 2], chi[1, 3]
        chi[0, 1], chi[2, 3], chi[1, 0], chi[3, 2] = 0, 0, 0, 0

    else:
        raise ValueError("Invalid phase. Phase must be 0, pi/2, pi, or 3*pi/2.")

    if representation == 'PTM':
        chi = chi2ptm(chi)

    return chi


def build_gate_ptm(params, pulse_duration, markovian_approx = True, phase = 0):

    """
    Builds a gate using the given parameters with the coloured phase noise parameterization
    directly in the PTM representation.

    Args:
        params (np.array): An array of parameters Gamma_1, Delta_1 (Gamma_2, Delta_2).
        pulse_duration (float): The duration of the pulse.
        p (float):Number of times the single gate is repeated in the long-sequence scheme.
        markovian_approx (bool, optional): Whether to use the Markovian approximation. Defaults to True.
        phase (float, optional): The phase value. Defaults to 0.

    Returns:
        numpy.ndarray: The gate matrix.

    Raises:
        ValueError: If an invalid phase value is provided.
    """

    #check whether to impose the markovian approximation
    if markovian_approx:
        params[2:] = 0

    #parameters
    Gamma_1, Delta_1, Gamma_2, Delta_2 = params[0], params[1], params[2], params[3]

    #theta
    theta = 1 / 2 * cmath.sqrt(Delta_1 ** 2 - Delta_2 ** 2 - Gamma_2 ** 2)

    #parts of matrix elements
    sigma_1 = np.real(np.exp(-Gamma_1 / 2) * (cmath.cos(theta) * np.cos(pulse_duration) \
                                      + Delta_1 * cmath.sin(theta) * np.sin(pulse_duration) / (2 *  theta) ))
    
    sigma_2 = - np.real(np.exp(-Gamma_1 / 2) * (cmath.cos(theta) * np.sin(pulse_duration) \
                                      + Delta_1 * cmath.sin(theta) * np.cos(pulse_duration) / (2 *  theta) ))

    
    sigma_3 = np.real(cmath.sin(theta) * np.exp(- Gamma_1 / 2) * (- Delta_2 * np.cos(pulse_duration) + \
                                                          Gamma_2 * np.sin(pulse_duration)) / (2 *  theta))
    #bulid the matrix
    chi = np.zeros((4, 4), 'd')
    chi[0,0] = 1

    if phase == 0:

        chi[1,1] = np.exp(-Gamma_1)
        chi[2,2] = sigma_1 - np.exp(- Gamma_1 / 2) * (cmath.sin(theta) / theta) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[3,3] = sigma_1 + np.exp(- Gamma_1 / 2) * (cmath.sin(theta) / theta) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[2,3] = sigma_3 - sigma_2
        chi[3,2] = sigma_3 + sigma_2

    elif phase == np.pi / 2:

        chi[2,2] = np.exp(-Gamma_1)
        chi[1,1] = sigma_1 - np.exp(- Gamma_1 / 2) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[3,3] = sigma_1 + np.exp(- Gamma_1 / 2) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[1,3] = sigma_3 - sigma_2
        chi[3,1] = sigma_3 + sigma_2

    elif phase == np.pi:

        chi[1,1] = np.exp(-Gamma_1)
        chi[2,2] = sigma_1 - np.exp(- Gamma_1 / 2) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[3,3] = sigma_1 + np.exp(- Gamma_1 / 2) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[2,3] = - sigma_3 + sigma_2
        chi[3,2] = - sigma_3 - sigma_2

    elif phase == 3 * np.pi / 2:

        chi[2,2] = np.exp(-Gamma_1)
        chi[1,1] = sigma_1 - np.exp(- Gamma_1 / 2) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[3,3] = sigma_1 + np.exp(- Gamma_1 / 2) * (Gamma_2 * np.cos(pulse_duration) + \
                                                          Delta_2 * np.sin(pulse_duration)) / 2
        chi[1,3] = - sigma_3 + sigma_2
        chi[3,1] = - sigma_3 - sigma_2

    return chi


def quantum_channel(pulse_duration, omega, tau, c, p = 1, markovian_approx = True, representation = 'PTM', phase = 0):
  
    """
    Calculates the quantum channel matrix based on the given parameters.

    Args:
        pulse_duration (float): The duration of the pulse.
        omega (float): The value of the Rabi frequency.
        tau (float): The value of the correlation time.
        c (float): The value of the diffusion constant.
        p (float, optional): The value of p. (Default is 1) (this is unused)
        markovian_approx (bool, optional): Whether to use the Markovian approximation or not. (Default is True)
        representation (str, optional): The representation of the quantum channel matrix. (Default is 'PTM')
        phase (float, optional): The phase value. (Default is 0)

    Returns:
        tuple: A tuple containing the quantum channel matrix (chi) and the values of Gamma1, Delta1, Gamma2, Delta2.
    """

    # Calculate Gamma1 and Delta1 for OU noise
    Gamma_1 = p * (1 / 2) * (c * tau ** 2) / (1 + (omega * tau) ** 2) * (pulse_duration / omega - tau * (2 * omega * tau) / (1 + (omega * tau) ** 2) \
        * exp(-(pulse_duration / omega) / tau) * sin(pulse_duration) - tau * (1 - (omega * tau) ** 2) / (1 + (omega * tau) ** 2) * (1 - exp(-(pulse_duration / omega) / tau) * cos(pulse_duration)))
    Delta_1 = p * (1 / 2) * (c * tau ** 2) / (1 + (omega * tau) ** 2) * (pulse_duration * tau + tau * (1 - (omega * tau) ** 2) / (1 + (omega * tau) ** 2) \
        * exp(-(pulse_duration / omega) / tau) * sin(pulse_duration) - tau * (2 * omega * tau) / (1 + (omega * tau) ** 2) * (1 - exp(-(pulse_duration / omega) / tau) * cos(pulse_duration)))

    #check wheter to impose the Markovian apprpoximation
    if markovian_approx:
        Gamma_2 = 0
        Delta_2 = 0
    else:
        # Calculate Gamma2 and Delta2
        Gamma_2 = p * (1 / 2) * (c * tau ** 2) / (1 + (omega * tau) ** 2) * cos(pulse_duration) * ((1 / omega) * sin(pulse_duration) - tau * cos(pulse_duration) + tau * exp(-(pulse_duration / omega) / tau))
        Delta_2 = p * (1 / 2) * (c * tau ** 2) / (1 + (omega * tau) ** 2) * ((1 / omega) * sin(pulse_duration) ** 2 - (tau / 2) * sin(2 * pulse_duration) + sin(pulse_duration) * tau * exp(-(pulse_duration / omega) / tau))

    gate = build_gate(np.array([Gamma_1, Delta_1, Gamma_2, Delta_2]), pulse_duration, p, markovian_approx, representation,  phase)
    # #Use build_gate_ptm if working with PTM representation for efficiency
    #gate = build_gate_ptm(np.array([Gamma_1, Delta_1, Gamma_2, Delta_2]), pulse_duration, markovian_approx , phase)

    # Divide by p to return the parameter values
    Gamma_1, Gamma_2, Delta_1, Delta_2 = Gamma_1 / p, Gamma_2 / p, Delta_1 / p, Delta_2 / p

    return gate, Gamma_1, Delta_1, Gamma_2, Delta_2
    

# The `gate_set` class is designed to represent a set of quantum gates along with their associated parameters and configurations.
class gate_set:

    def __init__(self, params, p, markovian_approx, representation, physical_gate_set_params):
        # `params` stores the parameters necessary for gate construction or operation.
        self.params = params
        # `p` denotes the long-sequence depth
        self.p = p
        # `markovian_approx` indicates whether a Markovian approximation is used.
        self.markovian_approx = markovian_approx
        # `representation` specifies the mathematical or physical representation of the gate (e.g., Pauli Transfer Matrix).
        self.representation = representation
        # `physical_gate_set_params` stores additional parameters describing the gate set.
        self.physical_gate_set_params = physical_gate_set_params

    # The `gates` method generates a list of gates based on the provided parameters and configurations.
    def gates(self):
        # Defines intervals for parameter selection, should be changed when using a different gate set.
        interval = [range(4), range(4,8), range(4,8), range(4,8), range(4,8)]
        # Constructs and returns a list of gates by iterating over `physical_gate_set_params` and applying the `build_gate` function.
        return [build_gate(self.params[interval[i]], self.physical_gate_set_params[i,0], self.p, self.markovian_approx, self.representation, self.physical_gate_set_params[i,1])
                for i in range(self.physical_gate_set_params.shape[0])] 
    
    # The `state` method returns the accesible state in the gate set.
    def state(self):
        # Constructs and returns a PTM state based on a subset of `params`.
        return ptm_state(self.params[8:11])

    # The `measurement` method returns the measurement operation.
    def measurement(self):
        # Constructs and returns a PTM state for measurement based on another subset of `params`.
        return ptm_state(self.params[11:15])

# The `gate_set_noise_params` class is analog to the `gate_set' class, but it provides a gate set based on the OU noise parameters,
# not the coloured phase noise parameters.
class gate_set_noise_params:

    def __init__(self, noise_params, state_params, measurement_params, p, markovian_approx, representation, physical_gate_set_params):
        self.noise_params = noise_params
        self.state_params = state_params
        self.measurement_params = measurement_params
        self.p = p
        self.markovian_approx = markovian_approx
        self.representation = representation
        self.physical_gate_set_params =  physical_gate_set_params

    def gates(self):
        return [quantum_channel(self.physical_gate_set_params[i,0], self.noise_params[0], self.noise_params[1],
                                self.noise_params[2], self.p, self.markovian_approx, self.representation,
                                self.physical_gate_set_params[i,1])[0]

                for i in range(self.physical_gate_set_params.shape[0])]

    def state(self):
        return ptm_state(self.state_params)

    def measurement(self):  
        return ptm_state(self.measurement_params)
    

def stochastic_matrices(phis, Mnoise, pulse_duration, c, tau, omega):

    """
    Calculate stochastic matrices for a quantum system under OU noise and operational conditions.

    This function computes the stochastic matrices under phase noise,
    considering the dynamics of the system over a given pulse duration.
   
    Parameters:
    - phis: Array of phase angles that define the desired gates.
    - Mnoise: The number of stochastic trajectories to average over.
    - pulse_duration: The duration of the pulse applied to the system, which defines the
                      time frame over which the system evolves.
    - c: difusion constant for the OU noise.
    - tau: correlation time for the OU noise
    - omega: The angular frequency of the system's evolution, defining how quickly the system
             oscillates as it evolves over time.

    Returns:
    - A tuple containing the stochastic matrices calculated for the system.
    """

    # Simulation time
    tf = pulse_duration / omega
    dt = tf / 1e2
    t = np.arange(0, tf, dt)  # sample points for the cosine Fourier transform
    N = len(t)

    # Spin operators
    I2 = np.eye(2)
    sz = np.array([[1, 0], [0, -1]])
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])

    #lambda matrix used in linear QPT
    lambda_matrix = 0.5 * np.block([[I2, sx], [sx, -I2]])

    # Orthogonal basis for the chi-matrix representation
    E = np.zeros((2, 2, 4), dtype=complex)
    E[:, :, 0] = I2
    E[:, :, 1] = sx
    E[:, :, 2] = -1j * sy
    E[:, :, 3] = sz

    # Basis operators under the quantum channel
    v0 = np.array([1, 0])
    v1 = np.array([0, 1])
    rho0 = np.zeros((2, 2, 4), dtype=complex)
    rho0[:, :, 0] = np.outer(v0, v0)
    rho0[:, :, 1] = np.outer(v0, v1)
    rho0[:, :, 2] = np.outer(v1, v0)
    rho0[:, :, 3] = np.outer(v1, v1)

    matrices = []

    # Precompute some values for efficiency
    exp_neg_dt_tau = np.exp(-dt / tau)
    sqrt_term = np.sqrt(0.5 * c * tau * (1 - np.exp(-2 * dt / tau)))

    for phi in phis:

        chi_avg = np.zeros((4, 4), dtype=complex)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # stochastic trajectories
        for _ in range(int(Mnoise)):
            Xn = np.zeros(N + 1)
            U = np.eye(2, dtype=complex)
            rho_p = np.copy(rho0)

            # Trotter evolution
            for k in range(N):
                w = np.random.normal()
                Xn[k + 1] = Xn[k] * exp_neg_dt_tau + sqrt_term * w
                U = expm(-1j * 0.5 * ((Xn[k]) * sz + omega * (cos_phi * sx - sin_phi * sy)) * dt) @ U

            for n in range(4):
                rho_p[:, :, n] = U @ rho_p[:, :, n] @ U.T.conj()

            rho_p_matrix = np.block([[rho_p[:, :, 0], rho_p[:, :, 1]], [rho_p[:, :, 2], rho_p[:, :, 3]]])

            # Chi-matrix representation of the dynamical quantum map using linear QPT
            chi_avg += lambda_matrix @ rho_p_matrix @ lambda_matrix / Mnoise

        # Basis change -i\sigma_y -> \sigma_y
        chi_avg[:, 2] *= 1j
        chi_avg[2, :] *= -1j

        matrices.append(chi_avg)

    return matrices

#Some functions to obtain long sequence base circuits
 
def repeat_central_element(circuit_list, p):

    repeated_circuit_list = circuit_list.copy()
    for i in range(1, p):
        repeated_circuit_list.insert(2, circuit_list[len(circuit_list) - 2])
    
    return repeated_circuit_list

def long_sequence_circuit_list(circuits_list, maxLengths):
    
    circuits_list_copy = []
    for p in maxLengths:
        for circuit_list in circuits_list:
            circuits_list_copy.append(pygsti.circuits.Circuit(repeat_central_element(circuit_list, p)))

    return circuits_list_copy
