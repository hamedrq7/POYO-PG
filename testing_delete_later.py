
import re
def regex_for_suffix(s2: str) -> str:
    return r".*" + re.escape(s2) + r"$"

def _generate_unit_mask(self, units: ArrayDict) -> np.ndarray:
        unit_mask = np.array([bool(self.pattern.search(uid)) for uid in units.id])
        if not self.keep_matches:
            unit_mask = ~unit_mask
        return unit_mask