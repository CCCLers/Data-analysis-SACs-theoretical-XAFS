import logging
import json
import os

logger = logging.getLogger(__name__)

class PresetsManager:
    def __init__(self, filename="xas_presets.json"):
        self.filename = self._get_settings_path(filename)
        self.presets = self._load_presets()

    def _get_settings_path(self, filename):
        # 确保预设文件保存在软件运行目录下
        base_dir = os.path.dirname(os.path.abspath(__file__))
        settings_dir = os.path.join(base_dir, 'settings')
        os.makedirs(settings_dir, exist_ok=True)
        return os.path.join(settings_dir, filename)

    def _load_presets(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading presets: {e}")
                return self._get_default_presets()
        else:
            return self._get_default_presets()

    def _get_default_presets(self):
        # 默认预设
        return {
            "Default - Fe K-edge": {
                "atom_kind": "Fe",
                "donor_type": "1s",
                "excitation_type": "2p",
                "emin": "700",
                "emax": "730",
                "npts": "2000",
                "broadening_regions_lorentz": "emin,emax,0.4",
                "broadening_regions_gauss": "emin,emax,1.0"
            }
        }

    def save_presets(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving presets: {e}")

    def add_preset(self, name, data):
        self.presets[name] = data
        self.save_presets()

    def delete_preset(self, name):
        if name in self.presets:
            del self.presets[name]
            self.save_presets()

    def get_preset_names(self):
        return list(self.presets.keys())

    def get_preset_data(self, name):
        return self.presets.get(name, {})