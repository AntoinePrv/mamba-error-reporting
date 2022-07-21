import libmambapy


def old_error_report(solver: libmambapy.Solver) -> str:
    message = ["Mamba failed to solve. The reported errors are:"]
    message += ["   " + l for l in solver.problems_to_str().split("\n")]
    return "\n".join(message)
