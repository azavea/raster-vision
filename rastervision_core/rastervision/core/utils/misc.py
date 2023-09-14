from pydantic import confloat

Proportion = confloat(ge=0, le=1)
