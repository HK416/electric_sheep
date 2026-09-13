//! Tokenizer and recursive-descent prim parser for the `.usda` subset documented in
//! `docs/api-notes/usd.md`.
//!
//! Hand-written on purpose: spec 1.9 makes the native USD reader cuttable, so it may not grow
//! the dependency graph, and the subset is small enough that a parser is cheaper than a
//! binding (spec 2 keeps the core pure Rust).

use std::collections::BTreeMap;

use es_math::Quat;

use crate::{Prim, Specifier, StageMeta, UpAxis, UsdError, UsdStage, Value};

/// Guard against a pathological file recursing the parser into a stack overflow.
const MAX_DEPTH: usize = 64;

/// Composition arcs and other layer features this reader refuses (api-note 5). Route: flatten
/// with USD Bake (spec 2.5).
const UNSUPPORTED_META: &[&str] = &[
    "references",
    "payload",
    "inherits",
    "specializes",
    "variantSets",
    "variants",
    "subLayers",
    "variantSelection",
];

/// Parses a `.usda` layer.
///
/// # Errors
///
/// [`UsdError::Syntax`] with the source line for a grammar violation, or
/// [`UsdError::Unsupported`] naming the prim for a feature outside the documented subset.
pub fn parse_usda(text: &str) -> Result<UsdStage, UsdError> {
    let body = strip_header(text)?;
    let tokens = lex(body.text, body.line)?;
    let mut parser = Parser {
        toks: tokens,
        pos: 0,
        warnings: Vec::new(),
    };
    let mut meta = StageMeta::default();
    if parser.peek_punct('(') {
        let entries = parser.meta_block("/")?;
        apply_layer_meta(&mut meta, &entries);
    }
    let mut prims = Vec::new();
    while parser.peek().is_some() {
        prims.push(parser.prim("", 0)?);
    }
    let mut warnings = parser.warnings;
    if !meta.up_axis_authored {
        warnings.push("layer has no upAxis; assuming USD's fallback `Y` (spec 3.1 is Z-up)".into());
    }
    if !meta.meters_per_unit_authored {
        warnings
            .push("layer has no metersPerUnit; assuming USD's fallback 0.01 (centimetres)".into());
    }
    Ok(UsdStage {
        meta,
        prims,
        warnings,
    })
}

struct Header<'a> {
    text: &'a str,
    line: usize,
}

/// Consumes the `#usda 1.0` magic and rejects the binary containers by their own magic.
fn strip_header(text: &str) -> Result<Header<'_>, UsdError> {
    if text.starts_with("PXR-USDC") {
        return Err(unsupported("/", ".usdc (binary crate file)", 1));
    }
    if text.starts_with("PK\u{3}\u{4}") {
        return Err(unsupported("/", ".usdz (zip container)", 1));
    }
    let mut line = 1;
    let mut rest = text;
    loop {
        let (head, tail) = rest.split_once('\n').unwrap_or((rest, ""));
        let trimmed = head.trim();
        if trimmed.starts_with("#usda") {
            return Ok(Header {
                text: tail,
                line: line + 1,
            });
        }
        if !trimmed.is_empty() {
            return Err(UsdError::Syntax {
                line,
                message: "expected a `#usda 1.0` header on the first non-blank line".into(),
            });
        }
        if tail.is_empty() {
            return Err(UsdError::Syntax {
                line,
                message: "empty layer: no `#usda 1.0` header".into(),
            });
        }
        rest = tail;
        line += 1;
    }
}

fn apply_layer_meta(meta: &mut StageMeta, entries: &[(String, Value)]) {
    for (key, value) in entries {
        match key.as_str() {
            "upAxis" => {
                if let Some(axis) = value.as_str() {
                    meta.up_axis = if axis.eq_ignore_ascii_case("z") {
                        UpAxis::Z
                    } else {
                        UpAxis::Y
                    };
                    meta.up_axis_authored = true;
                }
            }
            "metersPerUnit" => {
                if let Some(v) = value.as_f64() {
                    meta.meters_per_unit = v;
                    meta.meters_per_unit_authored = true;
                }
            }
            "defaultPrim" => meta.default_prim = value.as_str().map(str::to_owned),
            _ => {}
        }
    }
}

fn unsupported(path: &str, feature: &str, line: usize) -> UsdError {
    UsdError::Unsupported {
        path: path.to_owned(),
        feature: feature.to_owned(),
        line,
    }
}

// -------------------------------------------------------------------------------------------
// Tokenizer
// -------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Num(f64),
    /// `</World/base>`
    Path(String),
    /// `@./mesh.usda@`
    Asset(String),
    Punct(char),
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: usize,
}

fn syntax(line: usize, message: impl Into<String>) -> UsdError {
    UsdError::Syntax {
        line,
        message: message.into(),
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | ':' | '.')
}

fn lex(src: &str, start_line: usize) -> Result<Vec<Token>, UsdError> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = start_line;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            c if c.is_whitespace() => i += 1,
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '"' => {
                let (text, next, lines) = lex_string(&chars, i, line)?;
                out.push(Token {
                    tok: Tok::Str(text),
                    line,
                });
                line += lines;
                i = next;
            }
            '<' => {
                let (text, next) = lex_delimited(&chars, i + 1, '>', line, "prim path")?;
                out.push(Token {
                    tok: Tok::Path(text),
                    line,
                });
                i = next;
            }
            '@' => {
                let (text, next) = lex_delimited(&chars, i + 1, '@', line, "asset path")?;
                out.push(Token {
                    tok: Tok::Asset(text),
                    line,
                });
                i = next;
            }
            '{' | '}' | '(' | ')' | '[' | ']' | '=' | ',' | ';' => {
                out.push(Token {
                    tok: Tok::Punct(c),
                    line,
                });
                i += 1;
            }
            '-' | '+' | '.' | '0'..='9' => {
                let (value, next) = lex_number(&chars, i, line)?;
                out.push(Token {
                    tok: Tok::Num(value),
                    line,
                });
                i = next;
            }
            c if is_ident_start(c) => {
                let start = i;
                while i < chars.len() && is_ident_char(chars[i]) {
                    i += 1;
                }
                out.push(Token {
                    tok: Tok::Ident(chars[start..i].iter().collect()),
                    line,
                });
            }
            other => return Err(syntax(line, format!("unexpected character `{other}`"))),
        }
    }
    Ok(out)
}

/// Quoted string, single or triple. Returns the text, the index past the closing quote, and
/// how many newlines were consumed.
fn lex_string(chars: &[char], at: usize, line: usize) -> Result<(String, usize, usize), UsdError> {
    let triple = chars[at..].starts_with(&['"', '"', '"']);
    let quote_len = if triple { 3 } else { 1 };
    let mut i = at + quote_len;
    let mut text = String::new();
    let mut lines = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            text.push(chars[i + 1]);
            i += 2;
            continue;
        }
        if chars[i] == '"' && (!triple || chars[i..].starts_with(&['"', '"', '"'])) {
            return Ok((text, i + quote_len, lines));
        }
        if chars[i] == '\n' {
            if !triple {
                return Err(syntax(line, "unterminated string"));
            }
            lines += 1;
        }
        text.push(chars[i]);
        i += 1;
    }
    Err(syntax(line, "unterminated string"))
}

fn lex_delimited(
    chars: &[char],
    at: usize,
    close: char,
    line: usize,
    what: &str,
) -> Result<(String, usize), UsdError> {
    let mut i = at;
    let mut text = String::new();
    while i < chars.len() {
        if chars[i] == close {
            return Ok((text, i + 1));
        }
        if chars[i] == '\n' {
            break;
        }
        text.push(chars[i]);
        i += 1;
    }
    Err(syntax(line, format!("unterminated {what}")))
}

fn lex_number(chars: &[char], at: usize, line: usize) -> Result<(f64, usize), UsdError> {
    let mut i = at;
    if matches!(chars[i], '-' | '+') {
        i += 1;
    }
    let digits_start = i;
    while i < chars.len() {
        match chars[i] {
            '0'..='9' | '.' => i += 1,
            'e' | 'E' => {
                i += 1;
                if i < chars.len() && matches!(chars[i], '-' | '+') {
                    i += 1;
                }
            }
            _ => break,
        }
    }
    if i == digits_start {
        return Err(syntax(line, "expected a number"));
    }
    let text: String = chars[at..i].iter().collect();
    let value = text
        .parse::<f64>()
        .map_err(|_| syntax(line, format!("`{text}` is not a number")))?;
    Ok((value, i))
}

// -------------------------------------------------------------------------------------------
// Parser
// -------------------------------------------------------------------------------------------

/// An untyped literal, before the declared type name gives it a meaning.
#[derive(Clone, Debug)]
enum Raw {
    Num(f64),
    Str(String),
    Ident(String),
    Path(String),
    Asset(String),
    Tuple(Vec<Raw>),
    List(Vec<Raw>),
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    warnings: Vec<String>,
}

/// Qualifiers that may precede an attribute or relationship declaration.
const QUALIFIERS: &[&str] = &[
    "uniform", "custom", "config", "varying", "prepend", "append", "delete", "add", "reorder",
];

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }

    fn line(&self) -> usize {
        self.toks
            .get(self.pos)
            .or_else(|| self.toks.last())
            .map_or(1, |t| t.line)
    }

    fn next(&mut self) -> Option<Tok> {
        let tok = self.toks.get(self.pos).map(|t| t.tok.clone());
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn peek_punct(&self, c: char) -> bool {
        matches!(self.peek(), Some(Tok::Punct(p)) if *p == c)
    }

    fn eat_punct(&mut self, c: char) -> bool {
        let hit = self.peek_punct(c);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn expect_punct(&mut self, c: char) -> Result<(), UsdError> {
        if self.eat_punct(c) {
            Ok(())
        } else {
            Err(syntax(
                self.line(),
                format!("expected `{c}`, found {}", self.describe()),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, UsdError> {
        if let Some(Tok::Ident(name)) = self.peek() {
            let name = name.clone();
            self.pos += 1;
            return Ok(name);
        }
        Err(syntax(
            self.line(),
            format!("expected an identifier, found {}", self.describe()),
        ))
    }

    fn expect_str(&mut self) -> Result<String, UsdError> {
        if let Some(Tok::Str(name)) = self.peek() {
            let name = name.clone();
            self.pos += 1;
            return Ok(name);
        }
        Err(syntax(
            self.line(),
            format!("expected a quoted name, found {}", self.describe()),
        ))
    }

    fn describe(&self) -> String {
        match self.peek() {
            None => "end of file".to_owned(),
            Some(Tok::Ident(s)) => format!("`{s}`"),
            Some(Tok::Str(s)) => format!("string \"{s}\""),
            Some(Tok::Num(v)) => format!("number {v}"),
            Some(Tok::Path(p)) => format!("path <{p}>"),
            Some(Tok::Asset(a)) => format!("asset @{a}@"),
            Some(Tok::Punct(c)) => format!("`{c}`"),
        }
    }

    /// `( key = value ... )`. Returns the entries; rejects composition arcs.
    fn meta_block(&mut self, path: &str) -> Result<Vec<(String, Value)>, UsdError> {
        self.expect_punct('(')?;
        let mut entries = Vec::new();
        while !self.eat_punct(')') {
            if self.peek().is_none() {
                return Err(syntax(self.line(), "unterminated metadata block"));
            }
            if self.eat_punct(';') {
                continue;
            }
            // A bare documentation string.
            if matches!(self.peek(), Some(Tok::Str(_))) {
                self.pos += 1;
                continue;
            }
            let line = self.line();
            let mut key = self.expect_ident()?;
            while QUALIFIERS.contains(&key.as_str()) {
                key = self.expect_ident()?;
            }
            if UNSUPPORTED_META.contains(&key.as_str()) {
                return Err(unsupported(path, &key, line));
            }
            if !self.eat_punct('=') {
                continue;
            }
            let raw = self.raw_value(path)?;
            entries.push((key, coerce("", false, &raw, line)?));
        }
        Ok(entries)
    }

    /// `def|over|class [TypeName] "name" [( meta )] { body }`.
    fn prim(&mut self, parent_path: &str, depth: usize) -> Result<Prim, UsdError> {
        if depth > MAX_DEPTH {
            return Err(syntax(self.line(), "prim nesting is too deep"));
        }
        let line = self.line();
        let keyword = self.expect_ident()?;
        let specifier = match keyword.as_str() {
            "def" => Specifier::Def,
            "over" => Specifier::Over,
            "class" => Specifier::Class,
            other => {
                return Err(syntax(
                    line,
                    format!("expected `def`, `over` or `class`, found `{other}`"),
                ))
            }
        };
        let mut type_name = String::new();
        if let Some(Tok::Ident(_)) = self.peek() {
            type_name = self.expect_ident()?;
        }
        let name = self.expect_str()?;
        let path = format!("{parent_path}/{name}");

        let mut api_schemas = Vec::new();
        if self.peek_punct('(') {
            for (key, value) in self.meta_block(&path)? {
                if key == "apiSchemas" {
                    if let Some(list) = value.as_array() {
                        api_schemas
                            .extend(list.iter().filter_map(Value::as_str).map(str::to_owned));
                    }
                }
            }
        }

        let mut prim = Prim {
            name,
            specifier,
            type_name,
            attrs: BTreeMap::new(),
            rels: BTreeMap::new(),
            api_schemas,
            children: Vec::new(),
            line,
            path,
        };
        if self.eat_punct('{') {
            self.prim_body(&mut prim, depth)?;
        }
        if !KNOWN_TYPES.contains(&prim.type_name.as_str()) {
            self.warnings.push(format!(
                "{}: prim type `{}` has no mapping; kept, but it contributes nothing",
                prim.path, prim.type_name
            ));
        }
        Ok(prim)
    }

    fn prim_body(&mut self, prim: &mut Prim, depth: usize) -> Result<(), UsdError> {
        loop {
            if self.eat_punct('}') {
                return Ok(());
            }
            if self.peek().is_none() {
                return Err(syntax(
                    self.line(),
                    format!("`{}` is not closed", prim.path),
                ));
            }
            if self.eat_punct(';') {
                continue;
            }
            let line = self.line();
            let Some(Tok::Ident(first)) = self.peek().cloned() else {
                return Err(syntax(
                    line,
                    format!("expected a declaration, found {}", self.describe()),
                ));
            };
            match first.as_str() {
                "def" | "over" | "class" => {
                    let child = self.prim(&prim.path, depth + 1)?;
                    prim.children.push(child);
                }
                "variantSet" => return Err(unsupported(&prim.path, "variantSet", line)),
                _ => self.member(prim, line)?,
            }
        }
    }

    /// One attribute or relationship inside a prim body.
    fn member(&mut self, prim: &mut Prim, line: usize) -> Result<(), UsdError> {
        let mut word = self.expect_ident()?;
        while QUALIFIERS.contains(&word.as_str()) {
            word = self.expect_ident()?;
        }
        if word == "rel" {
            let name = self.expect_ident()?;
            let mut targets = Vec::new();
            if self.eat_punct('=') {
                match self.raw_value(&prim.path)? {
                    Raw::Path(p) => targets.push(p),
                    Raw::List(items) => {
                        for item in items {
                            if let Raw::Path(p) = item {
                                targets.push(p);
                            }
                        }
                    }
                    other => {
                        return Err(syntax(
                            line,
                            format!(
                                "relationship `{name}` needs a </path> target, found {other:?}"
                            ),
                        ))
                    }
                }
            }
            if self.peek_punct('(') {
                self.meta_block(&prim.path)?;
            }
            prim.rels.insert(name, targets);
            return Ok(());
        }

        // `word` is the type name; an array type is `word[]`.
        let is_array = if self.eat_punct('[') {
            self.expect_punct(']')?;
            true
        } else {
            false
        };
        let name = self.expect_ident()?;
        if let Some((_, suffix)) = name.rsplit_once('.') {
            if suffix == "timeSamples" {
                return Err(unsupported(&prim.path, "time samples", line));
            }
        }
        let mut value = None;
        if self.eat_punct('=') {
            let raw = self.raw_value(&prim.path)?;
            value = Some(coerce(&word, is_array, &raw, line)?);
        }
        if self.peek_punct('(') {
            self.meta_block(&prim.path)?;
        }
        if let Some(value) = value {
            prim.attrs.insert(name, value);
        }
        Ok(())
    }

    fn raw_value(&mut self, path: &str) -> Result<Raw, UsdError> {
        let line = self.line();
        match self.next() {
            Some(Tok::Num(v)) => Ok(Raw::Num(v)),
            Some(Tok::Str(s)) => Ok(Raw::Str(s)),
            Some(Tok::Ident(s)) => Ok(Raw::Ident(s)),
            Some(Tok::Path(p)) => Ok(Raw::Path(p)),
            Some(Tok::Asset(a)) => Ok(Raw::Asset(a)),
            Some(Tok::Punct('(')) => Ok(Raw::Tuple(self.raw_sequence(path, ')')?)),
            Some(Tok::Punct('[')) => Ok(Raw::List(self.raw_sequence(path, ']')?)),
            Some(Tok::Punct('{')) => Err(unsupported(path, "dictionary-valued attribute", line)),
            _ => Err(syntax(line, "expected a value")),
        }
    }

    fn raw_sequence(&mut self, path: &str, close: char) -> Result<Vec<Raw>, UsdError> {
        let mut items = Vec::new();
        loop {
            if self.eat_punct(close) {
                return Ok(items);
            }
            if self.peek().is_none() {
                return Err(syntax(self.line(), format!("expected `{close}`")));
            }
            items.push(self.raw_value(path)?);
            if !self.eat_punct(',') && !self.peek_punct(close) {
                return Err(syntax(
                    self.line(),
                    format!("expected `,` or `{close}`, found {}", self.describe()),
                ));
            }
        }
    }
}

/// Prim type names with a mapping; anything else is warned about but kept.
const KNOWN_TYPES: &[&str] = &[
    "",
    "Xform",
    "Scope",
    "Mesh",
    "Cube",
    "Sphere",
    "Cylinder",
    "Capsule",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "PhysicsFixedJoint",
    "PhysicsScene",
];

/// Type names whose value is a three-component vector.
const VEC3_TYPES: &[&str] = &[
    "float3",
    "double3",
    "half3",
    "int3",
    "point3f",
    "point3d",
    "vector3f",
    "vector3d",
    "normal3f",
    "color3f",
    "texCoord3f",
];

/// Gives a literal its declared meaning. `ty` empty means "no declared type": infer.
fn coerce(ty: &str, is_array: bool, raw: &Raw, line: usize) -> Result<Value, UsdError> {
    if is_array {
        let Raw::List(items) = raw else {
            return Err(syntax(line, format!("`{ty}[]` needs a `[...]` literal")));
        };
        return items
            .iter()
            .map(|item| coerce(ty, false, item, line))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array);
    }
    // An array literal under a scalar type name: keep it as an array anyway rather than lose it.
    if let Raw::List(items) = raw {
        return items
            .iter()
            .map(|item| coerce(ty, false, item, line))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array);
    }
    let wrong = |what: &str| syntax(line, format!("`{ty}` value must be {what}"));
    match ty {
        "bool" => match raw {
            Raw::Ident(s) if s == "true" => Ok(Value::Bool(true)),
            Raw::Ident(s) if s == "false" => Ok(Value::Bool(false)),
            Raw::Num(v) => Ok(Value::Bool(*v != 0.0)),
            _ => Err(wrong("`true` or `false`")),
        },
        "int" | "int64" | "uint" | "uint64" | "uchar" => match raw {
            Raw::Num(v) => Ok(Value::Int(*v as i64)),
            _ => Err(wrong("an integer")),
        },
        "float" | "half" => match raw {
            Raw::Num(v) => Ok(Value::Float(*v)),
            _ => Err(wrong("a number")),
        },
        "double" | "timecode" => match raw {
            Raw::Num(v) => Ok(Value::Double(*v)),
            _ => Err(wrong("a number")),
        },
        "token" => match raw {
            Raw::Str(s) | Raw::Ident(s) => Ok(Value::Token(s.clone())),
            _ => Err(wrong("a token")),
        },
        "string" => match raw {
            Raw::Str(s) | Raw::Ident(s) => Ok(Value::String(s.clone())),
            _ => Err(wrong("a string")),
        },
        "asset" => match raw {
            Raw::Asset(s) | Raw::Str(s) => Ok(Value::String(s.clone())),
            _ => Err(wrong("an @asset@ path")),
        },
        "quatf" | "quatd" | "quath" => match numbers(raw) {
            // The file writes (w, x, y, z) -- `GfQuatf(real, i, j, k)`; spec 3.1 stores xyzw.
            Some(n) if n.len() == 4 => Ok(Value::Quat(
                Quat::from_xyzw(n[1], n[2], n[3], n[0]).normalize(),
            )),
            _ => Err(wrong("a 4-component (w, x, y, z) tuple")),
        },
        "matrix4d" | "matrix4f" => match raw {
            Raw::Tuple(rows) if rows.len() == 4 => {
                let mut m = [[0.0f64; 4]; 4];
                for (out, row) in m.iter_mut().zip(rows) {
                    match numbers(row) {
                        Some(n) if n.len() == 4 => out.copy_from_slice(&n),
                        _ => return Err(wrong("four rows of four numbers")),
                    }
                }
                Ok(Value::Matrix4d(m))
            }
            _ => Err(wrong("four rows of four numbers")),
        },
        _ if VEC3_TYPES.contains(&ty) => match numbers(raw) {
            Some(n) if n.len() == 3 => Ok(Value::Float3([n[0], n[1], n[2]])),
            _ => Err(wrong("a 3-component tuple")),
        },
        // Unknown or undeclared type: preserve the literal as faithfully as we can.
        _ => Ok(match raw {
            Raw::Num(v) => Value::Double(*v),
            Raw::Str(s) => Value::String(s.clone()),
            Raw::Ident(s) => Value::Token(s.clone()),
            Raw::Path(p) => Value::Rel(p.clone()),
            Raw::Asset(a) => Value::String(a.clone()),
            Raw::Tuple(_) => match numbers(raw) {
                Some(n) if n.len() == 3 => Value::Float3([n[0], n[1], n[2]]),
                Some(n) => Value::Tuple(n),
                None => return Err(syntax(line, "tuple with a non-numeric component")),
            },
            Raw::List(_) => unreachable!("handled above"),
        }),
    }
}

/// A flat numeric tuple, or `None` if any component is not a number.
fn numbers(raw: &Raw) -> Option<Vec<f64>> {
    match raw {
        Raw::Tuple(items) => items
            .iter()
            .map(|i| match i {
                Raw::Num(v) => Some(*v),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_math::{Pose, Vec3};

    fn stage(body: &str) -> UsdStage {
        parse_usda(&format!("#usda 1.0\n{body}")).expect("fixture parses")
    }

    fn attr(body: &str, name: &str) -> Value {
        let s = stage(&format!("def Xform \"p\"\n{{\n{body}\n}}\n"));
        s.prims[0].attrs.get(name).expect("attribute").clone()
    }

    #[test]
    fn header_is_required_and_binary_formats_are_refused() {
        assert!(matches!(
            parse_usda("def Xform \"p\" {}"),
            Err(UsdError::Syntax { line: 1, .. })
        ));
        assert!(matches!(
            parse_usda("PXR-USDC\u{0}\u{0}"),
            Err(UsdError::Unsupported { .. })
        ));
    }

    #[test]
    fn layer_metadata_is_read_and_absence_is_warned() {
        let s = stage("(\n  defaultPrim = \"World\"\n  metersPerUnit = 0.01\n  upAxis = \"Y\"\n)\ndef Xform \"World\" {}\n");
        assert_eq!(s.meta.default_prim.as_deref(), Some("World"));
        assert!((s.meta.meters_per_unit - 0.01).abs() < 1e-12);
        assert_eq!(s.meta.up_axis, UpAxis::Y);
        assert!(s.warnings.is_empty());

        let bare = stage("def Xform \"World\" {}\n");
        assert_eq!(bare.meta.up_axis, UpAxis::Y);
        assert_eq!(bare.warnings.len(), 2);
    }

    #[test]
    fn every_value_kind_round_trips() {
        assert_eq!(
            attr(
                "bool physics:kinematicEnabled = true",
                "physics:kinematicEnabled"
            ),
            Value::Bool(true)
        );
        assert_eq!(
            attr("int faceVertexCounts = 3", "faceVertexCounts"),
            Value::Int(3)
        );
        assert_eq!(
            attr("float physics:mass = 2.5", "physics:mass"),
            Value::Float(2.5)
        );
        assert_eq!(attr("double size = -1e-3", "size"), Value::Double(-1e-3));
        assert_eq!(
            attr("uniform token axis = \"Z\"", "axis"),
            Value::Token("Z".into())
        );
        assert_eq!(
            attr("string doc = \"hi\"", "doc"),
            Value::String("hi".into())
        );
        assert_eq!(
            attr("asset file = @./m.usda@", "file"),
            Value::String("./m.usda".into())
        );
        assert_eq!(
            attr(
                "point3f physics:centerOfMass = (0, 1, 2)",
                "physics:centerOfMass"
            ),
            Value::Float3([0.0, 1.0, 2.0])
        );
        assert_eq!(
            attr(
                "matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (4,5,6,1) )",
                "xformOp:transform"
            ),
            Value::Matrix4d([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [4.0, 5.0, 6.0, 1.0]
            ])
        );
        assert_eq!(
            attr("int[] faceVertexIndices = [0, 1, 2]", "faceVertexIndices"),
            Value::Array(vec![Value::Int(0), Value::Int(1), Value::Int(2)])
        );
        assert_eq!(
            attr("float3[] points = [(0, 0, 0), (1, 0, 0)]", "points"),
            Value::Array(vec![
                Value::Float3([0.0; 3]),
                Value::Float3([1.0, 0.0, 0.0])
            ])
        );
        // A type this reader has no mapping for still survives as something.
        assert_eq!(
            attr("float2 uv = (1, 2)", "uv"),
            Value::Tuple(vec![1.0, 2.0])
        );
    }

    #[test]
    fn quat_literals_are_w_first_and_stored_xyzw() {
        // (w, x, y, z) = identity.
        assert_eq!(
            attr(
                "quatf physics:localRot0 = (1, 0, 0, 0)",
                "physics:localRot0"
            ),
            Value::Quat(Quat::IDENTITY)
        );
        // A 180 degree turn about x: (w, x, y, z) = (0, 1, 0, 0) -> xyzw (1, 0, 0, 0).
        let Value::Quat(q) = attr("quatf r = (0, 1, 0, 0)", "r") else {
            panic!("quat");
        };
        let turned = q.rotate(Vec3::new(0.0, 1.0, 0.0));
        assert!((turned.y + 1.0).abs() < 1e-12, "{turned:?}");
    }

    #[test]
    fn relationships_and_api_schemas_are_captured() {
        let s = stage(concat!(
            "def PhysicsRevoluteJoint \"j\" (\n",
            "    prepend apiSchemas = [\"PhysicsDriveAPI:angular\"]\n",
            ")\n{\n",
            "    rel physics:body0 = </World/a>\n",
            "    rel physics:body1 = [</World/b>]\n",
            "}\n"
        ));
        let prim = &s.prims[0];
        assert!(prim.has_api("PhysicsDriveAPI"));
        assert_eq!(prim.rels["physics:body0"], vec!["/World/a".to_owned()]);
        assert_eq!(prim.rels["physics:body1"], vec!["/World/b".to_owned()]);
    }

    #[test]
    fn nesting_builds_paths_and_children() {
        let s = stage(
            "def Xform \"World\"\n{\n  def Scope \"g\"\n  {\n    def Sphere \"s\" {}\n  }\n}\n",
        );
        assert_eq!(
            s.find("/World/g/s").map(|p| p.type_name.as_str()),
            Some("Sphere")
        );
        let mut count = 0;
        s.walk(&mut |_| count += 1);
        assert_eq!(count, 3);
    }

    #[test]
    fn composition_arcs_are_refused_with_the_prim_path() {
        let err = parse_usda(concat!(
            "#usda 1.0\n",
            "def Xform \"World\"\n{\n",
            "    def Xform \"ref\" (\n        prepend references = @./other.usda@</Thing>\n    )\n    {\n    }\n",
            "}\n"
        ))
        .expect_err("references must be refused");
        let UsdError::Unsupported { path, feature, .. } = err else {
            panic!("expected Unsupported, got {err:?}");
        };
        assert_eq!(path, "/World/ref");
        assert_eq!(feature, "references");
    }

    #[test]
    fn syntax_errors_carry_the_line() {
        let err = parse_usda("#usda 1.0\n\ndef Xform \"a\"\n{\n    float x = }\n}\n")
            .expect_err("malformed");
        assert!(matches!(err, UsdError::Syntax { line: 5, .. }), "{err:?}");
    }

    #[test]
    fn xform_ops_compose_in_listed_order() {
        let s = stage(concat!(
            "def Xform \"a\"\n{\n",
            "    double3 xformOp:translate = (1, 0, 0)\n",
            "    quatf xformOp:orient = (0.70710678118654752, 0, 0, 0.70710678118654752)\n",
            "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\"]\n",
            "    def Xform \"b\"\n    {\n",
            "        double3 xformOp:translate = (0, 1, 0)\n",
            "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n",
            "    }\n",
            "}\n"
        ));
        // `a` translates by +X then turns +90 deg about z, so `b`'s local +Y lands on world
        // -X and cancels the translation exactly.
        let pose = s.resolve_xform("/a/b");
        assert!(pose.position.x.abs() < 1e-9, "{pose:?}");
        assert!(pose.position.y.abs() < 1e-9, "{pose:?}");
        // Reversing the order is a different pose: the rotation would then not touch the
        // translation at all.
        assert_ne!(pose, Pose::new(Vec3::new(1.0, 1.0, 0.0), Quat::IDENTITY));
    }

    #[test]
    fn xform_ops_without_an_order_are_warned_not_applied() {
        let s = stage("def Xform \"a\"\n{\n    double3 xformOp:translate = (1, 2, 3)\n}\n");
        assert_eq!(s.resolve_xform("/a"), Pose::IDENTITY);
        let mut warnings = Vec::new();
        s.prims[0].local_xform(&mut warnings);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn a_transform_matrix_becomes_a_pose() {
        let s = stage(concat!(
            "def Xform \"a\"\n{\n",
            "    matrix4d xformOp:transform = ( (0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (7, 8, 9, 1) )\n",
            "    uniform token[] xformOpOrder = [\"xformOp:transform\"]\n",
            "}\n"
        ));
        let pose = s.resolve_xform("/a");
        assert_eq!(pose.position, Vec3::new(7.0, 8.0, 9.0));
        // The rows are the images of x and y: +X -> +Y, +Y -> -X, a +90 deg turn about z.
        let turned = pose.orientation.rotate(Vec3::new(1.0, 0.0, 0.0));
        assert!((turned.y - 1.0).abs() < 1e-9, "{turned:?}");
    }

    #[test]
    fn unknown_prim_types_and_attributes_survive_with_a_warning() {
        let s = stage("def DistantLight \"sun\"\n{\n    float inputs:intensity = 3000\n}\n");
        assert_eq!(s.prims[0].type_name, "DistantLight");
        assert!(s.prims[0].attrs.contains_key("inputs:intensity"));
        assert!(s.warnings.iter().any(|w| w.contains("DistantLight")));
    }

    #[test]
    fn deep_nesting_errors_instead_of_overflowing() {
        let mut text = String::from("#usda 1.0\n");
        for _ in 0..(MAX_DEPTH + 5) {
            text.push_str("def Xform \"p\"\n{\n");
        }
        for _ in 0..(MAX_DEPTH + 5) {
            text.push_str("}\n");
        }
        assert!(matches!(parse_usda(&text), Err(UsdError::Syntax { .. })));
    }
}
