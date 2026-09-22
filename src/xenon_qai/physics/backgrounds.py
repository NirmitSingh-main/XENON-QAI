"""
Background Source Physics and Characteristics for XENON Detectors.

Defines the physical characteristics of the 5 background classes in XENON-QAI:
- ER (Electronic Recoil): Beta/gamma events from 214Pb, 85Kr, material radioactivity
- RN (Radon Daughters): Alpha and beta decays from 222Rn / 220Rn decay chains
- WALL (Surface / PTFE): Events occurring near PTFE walls with partial charge loss
- AC (Accidental Coincidence): Uncorrelated lone S1 + lone S2 pairing
- CNNS (Coherent Neutrino-Nucleus Scattering): Irreducible solar/atmospheric neutrinos
"""

from typing import Dict, Any


BACKGROUND_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "ER": {
        "description": "Electronic Recoil from low-energy gammas and betas (e.g. 214Pb, 85Kr)",
        "recoil_type": "Electronic Recoil (ER)",
        "s2_over_s1_ratio": "High (large S2 ionization signal)",
        "spatial_distribution": "Homogeneous throughout active volume",
        "primary_discriminator": "log10(S2/S1) discrimination band",
    },
    "RN": {
        "description": "Radon daughter decays (222Rn/220Rn chain, 218Po, 214Pb, 214Bi-Po)",
        "recoil_type": "Mixed ER and high-energy alpha decays",
        "s2_over_s1_ratio": "Variable (alpha recoils have low S2/S1, beta has normal ER S2/S1)",
        "spatial_distribution": "Bulk detector volume with convection flow",
        "primary_discriminator": "Energy spectrum and delayed coincidence tagging (Bi-Po)",
    },
    "WALL": {
        "description": "Surface events near PTFE reflector walls suffering charge collection loss",
        "recoil_type": "Surface ER / alpha with severe S2 suppression",
        "s2_over_s1_ratio": "Suppressed (charges trapped or recombined at PTFE wall)",
        "spatial_distribution": "High radial coordinate R near wall radius R_max",
        "primary_discriminator": "Radial position R and top/bottom PMT pattern localization",
    },
    "AC": {
        "description": "Accidental Coincidence of uncorrelated isolated S1 and isolated S2 pulses",
        "recoil_type": "Non-physical pairing (no true single physical interaction)",
        "s2_over_s1_ratio": "Random, non-physical",
        "spatial_distribution": "Random drift times, unphysical position consistency",
        "primary_discriminator": "Drift time vs pulse width correlation and S1/S2 PMT hit pattern mismatch",
    },
    "CNNS": {
        "description": "Coherent Elastic Neutrino-Nucleus Scattering from 8B solar neutrinos",
        "recoil_type": "Nuclear Recoil (NR) — identical event topology to light WIMP scattering",
        "s2_over_s1_ratio": "Low (falls perfectly within NR signal band)",
        "spatial_distribution": "Completely uniform throughout detector",
        "primary_discriminator": "Energy spectrum (confined to < 4 keV_nr) and known solar flux expectation",
    }
}


def get_background_info(bkg_name: str) -> Dict[str, Any]:
    """Returns physics metadata for a specific background class."""
    return BACKGROUND_PROPERTIES.get(bkg_name.upper(), {
        "description": f"Unknown background class: {bkg_name}",
        "recoil_type": "Unknown",
        "s2_over_s1_ratio": "Unknown",
        "spatial_distribution": "Unknown",
        "primary_discriminator": "Unknown"
    })
