//! Pi coding agent provider registry.
//!
//! Maps provider names to the host environment variable that carries their
//! API key.  This is the single source of truth used for both credential
//! injection into containers and auto-detection of available providers.

/// A provider whose API key was found on the host.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DetectedProvider {
    /// Provider name (e.g. "anthropic", "openai").
    pub provider: String,
    /// Whether this is the configured default provider.
    pub is_default: bool,
}

/// Known Pi providers and the host env var that carries their API key.
///
/// `"google"` and `"gemini"` are aliases that share the same key; both are
/// listed so that [`provider_env_key`] resolves either spelling.
const PROVIDERS: &[(&str, &str)] = &[
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("google", "GEMINI_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("xai", "XAI_API_KEY"),
];

/// De-duplicated canonical list (excludes the `"gemini"` alias).
const CANONICAL_PROVIDERS: &[(&str, &str)] = &[
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("google", "GEMINI_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("xai", "XAI_API_KEY"),
];

/// Return the env-var name for a Pi provider, or `None` if unknown.
///
/// # Examples
/// ```
/// # use ironclaw::orchestrator::providers::provider_env_key;
/// assert_eq!(provider_env_key("anthropic"), Some("ANTHROPIC_API_KEY"));
/// assert_eq!(provider_env_key("gemini"), Some("GEMINI_API_KEY"));
/// assert_eq!(provider_env_key("google"), Some("GEMINI_API_KEY"));
/// assert_eq!(provider_env_key("unknown"), None);
/// ```
pub fn provider_env_key(provider: &str) -> Option<&'static str> {
    PROVIDERS
        .iter()
        .find(|(name, _)| *name == provider)
        .map(|(_, key)| *key)
}

/// Return the list of all supported provider names (canonical, no aliases).
pub fn supported_providers() -> &'static [(&'static str, &'static str)] {
    CANONICAL_PROVIDERS
}

/// Detect which Pi providers have API keys set on the host.
///
/// Returns a [`DetectedProvider`] for each canonical provider whose env var
/// is set and non-empty.  The `is_default` flag is set for the entry whose
/// name matches `default_provider`.
pub fn detect_providers(default_provider: &str) -> Vec<DetectedProvider> {
    detect_providers_with(default_provider, |key| {
        std::env::var(key).ok().filter(|v| !v.is_empty())
    })
}

/// Testable core of [`detect_providers`].
///
/// `lookup` is called with each env-var name; return `Some(non_empty_value)`
/// to indicate the key is present, or `None` to skip it.
fn detect_providers_with<F>(default_provider: &str, lookup: F) -> Vec<DetectedProvider>
where
    F: Fn(&str) -> Option<String>,
{
    CANONICAL_PROVIDERS
        .iter()
        .filter(|(_, env_key)| lookup(env_key).is_some())
        .map(|(name, _)| DetectedProvider {
            provider: name.to_string(),
            is_default: *name == default_provider,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Helper: build a lookup function from a set of env-var entries.
    fn env_from<'a>(
        entries: &'a [(&'a str, &'a str)],
    ) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = entries.iter().copied().collect();
        move |key: &str| {
            map.get(key)
                .map(|v| v.to_string())
                .filter(|v| !v.is_empty())
        }
    }

    // ── provider_env_key ──────────────────────────────────────────

    #[test]
    fn known_providers_resolve() {
        assert_eq!(provider_env_key("anthropic"), Some("ANTHROPIC_API_KEY"));
        assert_eq!(provider_env_key("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(provider_env_key("google"), Some("GEMINI_API_KEY"));
        assert_eq!(provider_env_key("groq"), Some("GROQ_API_KEY"));
        assert_eq!(provider_env_key("xai"), Some("XAI_API_KEY"));
    }

    #[test]
    fn gemini_alias_resolves() {
        assert_eq!(provider_env_key("gemini"), Some("GEMINI_API_KEY"));
        assert_eq!(provider_env_key("google"), provider_env_key("gemini"));
    }

    #[test]
    fn unknown_provider_returns_none() {
        assert_eq!(provider_env_key(""), None);
        assert_eq!(provider_env_key("azure"), None);
        assert_eq!(provider_env_key("ANTHROPIC"), None); // case-sensitive
        assert_eq!(provider_env_key("Openai"), None);
    }

    // ── supported_providers ───────────────────────────────────────

    #[test]
    fn supported_providers_no_duplicates() {
        let names: Vec<&str> = supported_providers().iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len());
    }

    #[test]
    fn supported_providers_contains_all_canonical() {
        let names: Vec<&str> = supported_providers().iter().map(|(n, _)| *n).collect();
        for expected in ["anthropic", "openai", "google", "groq", "xai"] {
            assert!(names.contains(&expected), "missing {expected}");
        }
        // "gemini" is an alias, should NOT appear in canonical list.
        assert!(!names.contains(&"gemini"));
    }

    // ── detect_providers_with ─────────────────────────────────────

    #[test]
    fn detect_with_no_keys() {
        let lookup = env_from(&[]);
        let detected = detect_providers_with("anthropic", lookup);
        assert!(detected.is_empty());
    }

    #[test]
    fn detect_with_one_key() {
        let lookup = env_from(&[("OPENAI_API_KEY", "sk-test-123")]);
        let detected = detect_providers_with("openai", lookup);
        assert_eq!(detected.len(), 1);
        assert_eq!(detected[0].provider, "openai");
        assert!(detected[0].is_default);
    }

    #[test]
    fn detect_marks_default_correctly() {
        let lookup = env_from(&[
            ("ANTHROPIC_API_KEY", "sk-ant-test"),
            ("OPENAI_API_KEY", "sk-test"),
        ]);
        let detected = detect_providers_with("anthropic", lookup);
        assert_eq!(detected.len(), 2);

        let anthropic = detected.iter().find(|p| p.provider == "anthropic").unwrap();
        assert!(anthropic.is_default);

        let openai = detected.iter().find(|p| p.provider == "openai").unwrap();
        assert!(!openai.is_default);
    }

    #[test]
    fn detect_skips_empty_values() {
        let lookup = env_from(&[("GROQ_API_KEY", ""), ("XAI_API_KEY", "xai-test")]);
        let detected = detect_providers_with("xai", lookup);
        assert_eq!(detected.len(), 1);
        assert_eq!(detected[0].provider, "xai");
    }

    #[test]
    fn detect_no_gemini_duplicate() {
        let lookup = env_from(&[("GEMINI_API_KEY", "gem-test")]);
        let detected = detect_providers_with("google", lookup);
        // Only "google" should appear, not "gemini".
        assert_eq!(detected.len(), 1);
        assert_eq!(detected[0].provider, "google");
        assert!(detected[0].is_default);
    }

    #[test]
    fn detect_all_providers_present() {
        let lookup = env_from(&[
            ("ANTHROPIC_API_KEY", "key1"),
            ("OPENAI_API_KEY", "key2"),
            ("GEMINI_API_KEY", "key3"),
            ("GROQ_API_KEY", "key4"),
            ("XAI_API_KEY", "key5"),
        ]);
        let detected = detect_providers_with("groq", lookup);
        assert_eq!(detected.len(), 5);

        let names: Vec<&str> = detected.iter().map(|p| p.provider.as_str()).collect();
        for expected in ["anthropic", "openai", "google", "groq", "xai"] {
            assert!(names.contains(&expected), "missing {expected}");
        }

        let groq = detected.iter().find(|p| p.provider == "groq").unwrap();
        assert!(groq.is_default);

        // All others should not be default.
        for p in &detected {
            if p.provider != "groq" {
                assert!(!p.is_default, "{} should not be default", p.provider);
            }
        }
    }

    #[test]
    fn detect_default_not_in_detected_list() {
        // Default is "anthropic" but only openai key is set.
        let lookup = env_from(&[("OPENAI_API_KEY", "sk-test")]);
        let detected = detect_providers_with("anthropic", lookup);
        assert_eq!(detected.len(), 1);
        assert_eq!(detected[0].provider, "openai");
        assert!(!detected[0].is_default);
    }

    // ── DetectedProvider serialization ────────────────────────────

    #[test]
    fn detected_provider_serializes_correctly() {
        let p = DetectedProvider {
            provider: "anthropic".to_string(),
            is_default: true,
        };
        let json = serde_json::to_value(&p).unwrap();
        assert_eq!(json["provider"], "anthropic");
        assert_eq!(json["is_default"], true);
    }

    #[test]
    fn detected_provider_equality() {
        let a = DetectedProvider {
            provider: "openai".to_string(),
            is_default: false,
        };
        let b = a.clone();
        assert_eq!(a, b);

        let c = DetectedProvider {
            provider: "openai".to_string(),
            is_default: true,
        };
        assert_ne!(a, c);
    }

    // ── consistency checks ────────────────────────────────────────

    #[test]
    fn canonical_is_subset_of_providers() {
        for (name, key) in CANONICAL_PROVIDERS {
            assert!(
                PROVIDERS.iter().any(|(n, k)| n == name && k == key),
                "canonical ({name}, {key}) not in PROVIDERS",
            );
        }
    }

    #[test]
    fn every_provider_env_key_has_canonical_entry() {
        for (_, key) in PROVIDERS {
            assert!(
                CANONICAL_PROVIDERS.iter().any(|(_, k)| k == key),
                "env key {key} from PROVIDERS not in CANONICAL_PROVIDERS",
            );
        }
    }

    #[test]
    fn provider_env_key_covers_all_entries() {
        // Every entry in PROVIDERS should be discoverable via provider_env_key.
        for (name, expected_key) in PROVIDERS {
            assert_eq!(
                provider_env_key(name),
                Some(*expected_key),
                "provider_env_key({name}) mismatch",
            );
        }
    }
}
