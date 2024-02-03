import pytest
import sys
sys.path.append('..')
from main import main




@pytest.mark.parametrize(
    "gradient_steps,substeps,gamma,feedback,control,goal,measure_op,max_steps,mode,type_input,input_type,"
    "n_average,kind_state_output",
    [
        (10, 1, 0.0, True, True, "fidelity", "both", 1, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (1, 1, 0.0, True, True, "fidelity", "both", 1, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "purity", "both", 1, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "fidelity", "both", 1, "network", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "fidelity", "both", 1, "network", "measure", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "fidelity", "both", 1, "lookup", "measure", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, False, True, "fidelity", "both", 1, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, False, "fidelity", "both", 1, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, False, False, "fidelity", "both", 1, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "fidelity", "both", 3, "lookup", "state", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "fidelity", "both", 3, "lookup", "memory", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.0, True, True, "fidelity", "both", 3, "lookup", "table", "thermal", 1, "four_legged_kitten"),
        (10, 1, 0.01, True, True, "fidelity", "both", 3, "lookup", "table", "thermal", 1, "four_legged_kitten"),
    ])
def test_technical(gradient_steps, substeps, gamma, feedback, control, goal, measure_op, max_steps, mode, type_input,
                   input_type, n_average, kind_state_output):
    assert main(
        gradient_steps=gradient_steps,
        substeps=substeps,
        gamma=gamma,
        feedback=feedback,
        control=control,
        goal=goal,
        measure_op=measure_op,
        max_steps=max_steps,
        mode=mode, type_input=type_input,
        input=input_type,
        n_average=n_average,
        kind_state_output=kind_state_output,
        return_fidelity=False
    )


@pytest.mark.parametrize(
    "gradient_steps,substeps,gamma,feedback,control,goal,measure_op,max_steps,mode,type_input,input_type,"
    "n_average,kind_state_output",
    [
        (100, 1, 0.0, False, True, "fidelity", "both", 4, "network", "state", None, 1, "fock"),
    ])
def test_convergence(gradient_steps, substeps, gamma, feedback, control, goal, measure_op, max_steps, mode, type_input,
                     input_type, n_average, kind_state_output):
    assert main(
        gradient_steps=gradient_steps,
        substeps=substeps,
        gamma=gamma,
        feedback=feedback,
        control=control,
        goal=goal,
        measure_op=measure_op,
        max_steps=max_steps,
        mode=mode, type_input=type_input,
        input=input_type,
        n_average=n_average,
        kind_state_output=kind_state_output,
        return_fidelity=True
    ) > 0.95


@pytest.mark.parametrize(
    "gradient_steps,substeps,gamma,feedback,control,goal,measure_op,max_steps,mode,type_input,input_type,"
    "n_average,kind_state_output,type_unitary,complex_fields,double_measure",
    [
        (10, 1, 0.0, False, True, "fidelity", "non-demolition", 9, "network", "measure", None, 1, "GKP", "SNAP", True,
         True),
    ])
def test_snap(gradient_steps, substeps, gamma, feedback, control, goal, measure_op, max_steps, mode, type_input,
              input_type, n_average, kind_state_output, type_unitary, complex_fields, double_measure):
    assert main(
        gradient_steps=gradient_steps,
        substeps=substeps,
        gamma=gamma,
        feedback=feedback,
        control=control,
        goal=goal,
        measure_op=measure_op,
        max_steps=max_steps,
        mode=mode, type_input=type_input,
        input="SNAP",
        n_average=n_average,
        kind_state_output=kind_state_output,
        return_fidelity=True,
        type_unitary=type_unitary,
        complex_fields=complex_fields,
        double_measure=double_measure,
        clock=True,
        N_snap=100,
        N_cavity=130,
        batch_size=2
    )


def test_qubits():
    assert main(
        gradient_steps=10,
        substeps=1,
        gamma=0.0,
        feedback=True,
        control=True,
        goal="fidelity",
        measure_op="both",
        max_steps=3,
        mode="network", type_input="measure",
        input="qubits",
        kind_state_output="qubits",
        return_fidelity=True,
        complex_fields=False,
        double_measure=True,
        clock=False,
        N_cavity=2,
        batch_size=2,
        system="qubits"
    )
