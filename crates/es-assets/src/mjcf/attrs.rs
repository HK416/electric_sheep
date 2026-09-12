//! Attribute access for one MJCF element: `<default>` class values merged under the explicit
//! ones, typed parsing with line-numbered errors, and unknown-attribute tracking.
//!
//! Every getter records the name it was asked for, so [`Attrs::report_unknown`] can name the
//! attributes the parser never looked at instead of dropping them silently.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use es_math::Vec3;
use roxmltree::Node;

use super::{MjcfError, Warning};

/// Merged attribute view of one element: explicit attributes win over class defaults.
#[derive(Debug)]
pub(crate) struct Attrs<'a> {
    node: Node<'a, 'a>,
    tag: &'a str,
    line: u32,
    defaults: BTreeMap<&'a str, &'a str>,
    asked: RefCell<BTreeSet<&'static str>>,
}

impl<'a> Attrs<'a> {
    pub(crate) fn new(
        node: Node<'a, 'a>,
        tag: &'a str,
        line: u32,
        defaults: BTreeMap<&'a str, &'a str>,
    ) -> Self {
        Self {
            node,
            tag,
            line,
            defaults,
            asked: RefCell::new(BTreeSet::new()),
        }
    }

    pub(crate) fn line(&self) -> u32 {
        self.line
    }

    pub(crate) fn tag(&self) -> &'a str {
        self.tag
    }

    /// Explicit attribute, else the value inherited from the element's default class.
    pub(crate) fn get(&self, name: &'static str) -> Option<&'a str> {
        self.asked.borrow_mut().insert(name);
        self.node
            .attribute(name)
            .or_else(|| self.defaults.get(name).copied())
    }

    pub(crate) fn get_or(&self, name: &'static str, default: &'a str) -> &'a str {
        self.get(name).unwrap_or(default)
    }

    /// Explicit attribute only, ignoring defaults — for `class` and `childclass`, which select
    /// a default class and so cannot themselves come from one.
    pub(crate) fn own(&self, name: &'static str) -> Option<&'a str> {
        self.asked.borrow_mut().insert(name);
        self.node.attribute(name)
    }

    pub(crate) fn bad(&self, attr: &'static str, value: &str) -> MjcfError {
        MjcfError::BadAttr {
            line: self.line,
            element: self.tag.to_owned(),
            attr,
            value: value.to_owned(),
        }
    }

    pub(crate) fn missing(&self, attr: &'static str) -> MjcfError {
        MjcfError::MissingAttr {
            line: self.line,
            element: self.tag.to_owned(),
            attr,
        }
    }

    /// MJCF booleans are the literals `true` / `false`.
    pub(crate) fn flag(&self, name: &'static str) -> Result<Option<bool>, MjcfError> {
        match self.get(name) {
            None => Ok(None),
            Some("true") => Ok(Some(true)),
            Some("false") => Ok(Some(false)),
            Some(other) => Err(self.bad(name, other)),
        }
    }

    pub(crate) fn flag_or(&self, name: &'static str, default: bool) -> Result<bool, MjcfError> {
        Ok(self.flag(name)?.unwrap_or(default))
    }

    /// MJCF tri-state (`true` / `false` / `auto`): `None` is `auto`.
    pub(crate) fn tristate(&self, name: &'static str) -> Result<Option<bool>, MjcfError> {
        match self.get(name) {
            None | Some("auto") => Ok(None),
            Some("true") => Ok(Some(true)),
            Some("false") => Ok(Some(false)),
            Some(other) => Err(self.bad(name, other)),
        }
    }

    pub(crate) fn num(&self, name: &'static str) -> Result<Option<f64>, MjcfError> {
        match self.get(name) {
            None => Ok(None),
            Some(text) => text
                .trim()
                .parse::<f64>()
                .map(Some)
                .map_err(|_| self.bad(name, text)),
        }
    }

    pub(crate) fn num_or(&self, name: &'static str, default: f64) -> Result<f64, MjcfError> {
        Ok(self.num(name)?.unwrap_or(default))
    }

    pub(crate) fn int_or(&self, name: &'static str, default: u32) -> Result<u32, MjcfError> {
        match self.get(name) {
            None => Ok(default),
            Some(text) => text
                .trim()
                .parse::<u32>()
                .map_err(|_| self.bad(name, text))
                .or_else(|e| {
                    // MJCF writes some integer attributes as `1.0`; accept that, reject the rest.
                    let v = text.trim().parse::<f64>().map_err(|_| e)?;
                    if v >= 0.0 && v.fract() == 0.0 {
                        Ok(v as u32)
                    } else {
                        Err(self.bad(name, text))
                    }
                }),
        }
    }

    /// Whitespace-separated numbers, any count.
    pub(crate) fn nums(&self, name: &'static str) -> Result<Option<Vec<f64>>, MjcfError> {
        let Some(text) = self.get(name) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for token in text.split_whitespace() {
            out.push(token.parse::<f64>().map_err(|_| self.bad(name, text))?);
        }
        Ok(Some(out))
    }

    /// Exactly `N` numbers, or an error naming the attribute.
    pub(crate) fn fixed<const N: usize>(
        &self,
        name: &'static str,
    ) -> Result<Option<[f64; N]>, MjcfError> {
        let Some(values) = self.nums(name)? else {
            return Ok(None);
        };
        <[f64; N]>::try_from(values.as_slice())
            .map(Some)
            .map_err(|_| self.bad(name, self.get(name).unwrap_or_default()))
    }

    /// Up to `N` numbers; the tail keeps the value it has in `default`.
    pub(crate) fn padded<const N: usize>(
        &self,
        name: &'static str,
        default: [f64; N],
    ) -> Result<[f64; N], MjcfError> {
        let Some(values) = self.nums(name)? else {
            return Ok(default);
        };
        if values.len() > N {
            return Err(self.bad(name, self.get(name).unwrap_or_default()));
        }
        let mut out = default;
        out[..values.len()].copy_from_slice(&values);
        Ok(out)
    }

    pub(crate) fn vec3(&self, name: &'static str, default: Vec3) -> Result<Vec3, MjcfError> {
        match self.fixed::<3>(name)? {
            None => Ok(default),
            Some([x, y, z]) => Ok(Vec3::new(x, y, z)),
        }
    }

    pub(crate) fn range(&self, name: &'static str) -> Result<Option<(f64, f64)>, MjcfError> {
        Ok(self.fixed::<2>(name)?.map(|[lo, hi]| (lo, hi)))
    }

    /// Warn about every attribute the parser never asked for (spec 1.4: nothing silently
    /// dropped). Call once per element, after all getters.
    pub(crate) fn report_unknown(&self, warnings: &mut Vec<Warning>) {
        let asked = self.asked.borrow();
        for attr in self.node.attributes() {
            if !asked.contains(attr.name()) {
                warnings.push(Warning {
                    line: self.line,
                    message: format!(
                        "<{}>: attribute `{}` is not represented in SceneDesc",
                        self.tag,
                        attr.name()
                    ),
                });
            }
        }
        for name in self.defaults.keys() {
            if !asked.contains(*name) {
                warnings.push(Warning {
                    line: self.line,
                    message: format!(
                        "<{}>: default-class attribute `{name}` is not represented in SceneDesc",
                        self.tag
                    ),
                });
            }
        }
    }
}
