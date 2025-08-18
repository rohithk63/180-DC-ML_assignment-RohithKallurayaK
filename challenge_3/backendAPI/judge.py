def judge_decision(request):
    if request.winner:
        return {"verdict": f"Judge declares {request.winner} the winner!"}
    elif request.new_evidence:
        return {"verdict": f"New evidence introduced: {request.new_evidence}"}
    else:
        return {"verdict": "No decision made yet."}
