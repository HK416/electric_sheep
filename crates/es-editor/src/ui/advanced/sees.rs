//! What the policy sees (spec 7): each image of the opened bundle before and after the
//! Observation IR's preprocessing.

use eframe::egui;

use super::rgb_texture;
use crate::app::EditorApp;
use crate::model::i18n;
use crate::model::image_view::ImagePair;

impl EditorApp {
    pub(crate) fn images_tab(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let empty = self.t("sees.empty");
        let (before, after) = (self.t("sees.before"), self.t("sees.after"));
        let count = |pairs: usize, outputs: usize| {
            i18n::fill(
                self.settings.lang,
                "sees.count",
                &[&pairs.to_string(), &outputs.to_string()],
            )
        };
        let Some(opened) = &mut self.opened else {
            ui.label(empty);
            return;
        };
        if let Some(err) = &opened.image_error {
            ui.label(err.as_str());
        }
        let outputs = opened.observation.outputs.len();
        ui.label(count(opened.pairs.len(), outputs));
        for pair in &opened.pairs {
            let (pair_before, pair_after) = opened
                .textures
                .entry(pair.name.clone())
                .or_insert_with(|| (texture(&ctx, pair, true), texture(&ctx, pair, false)));
            ui.heading(&pair.name);
            ui.horizontal(|ui| {
                for (label, tex) in [(before, &*pair_before), (after, &*pair_after)] {
                    ui.vertical(|ui| {
                        ui.label(label);
                        let scale = (240.0 / tex.size_vec2().x).max(1.0);
                        ui.image(egui::load::SizedTexture::new(
                            tex.id(),
                            tex.size_vec2() * scale,
                        ));
                    });
                }
            });
            ui.separator();
        }
    }
}

fn texture(ctx: &egui::Context, pair: &ImagePair, before: bool) -> egui::TextureHandle {
    let img = if before { &pair.before } else { &pair.after };
    let name = format!("{}-{}", pair.name, if before { "before" } else { "after" });
    rgb_texture(ctx, &name, img)
}
