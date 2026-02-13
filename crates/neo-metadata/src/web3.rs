//! Web3 rights and royalty metadata for NEO files.
//!
//! Provides on-chain royalty split definitions and smart contract references.
//! See SPECIFICATION.md Section 9.4.
//!
//! # Security Note
//!
//! Wallet addresses and contract references are **informational only**.
//! Applications MUST NOT auto-execute transactions based on this metadata.

use serde::{Deserialize, Serialize};

use crate::error::{MetadataError, Result};

/// Known blockchain identifiers for royalty splits.
const SUPPORTED_CHAINS: &[&str] = &[
    "ethereum", "polygon", "solana", "tezos", "arbitrum", "optimism", "base",
];

/// Web3 rights and royalty information for a NEO file.
///
/// Contains smart contract references and royalty split definitions
/// for on-chain rights management.
///
/// # Example
/// ```
/// use neo_metadata::{RightsInfo, RoyaltySplit};
///
/// let rights = RightsInfo::new()
///     .with_contract("0x1234...abcd", "ethereum")
///     .with_license_uri("https://creativecommons.org/licenses/by/4.0/")
///     .with_split(RoyaltySplit::new("0xAAAA...1111", "ethereum", 0.7).with_name("Producer"))
///     .with_split(RoyaltySplit::new("0xBBBB...2222", "ethereum", 0.3).with_name("Vocalist"));
///
/// rights.validate().unwrap();
/// let json = rights.to_json_pretty().unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RightsInfo {
    /// Smart contract address managing rights (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract_address: Option<String>,

    /// Blockchain where the contract is deployed (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract_chain: Option<String>,

    /// Royalty split definitions.
    #[serde(default)]
    pub splits: Vec<RoyaltySplit>,

    /// URI to the license terms (e.g., Creative Commons URL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license_uri: Option<String>,
}

/// A single royalty split entry defining a payee's share.
///
/// Shares are expressed as fractions of 1.0 (e.g., 0.5 = 50%).
/// The sum of all shares in a `RightsInfo` MUST equal 1.0 (within tolerance).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoyaltySplit {
    /// Wallet address of the payee.
    pub address: String,

    /// Blockchain identifier (e.g., "ethereum", "polygon", "solana").
    pub chain: String,

    /// Fraction of royalties (0.0 to 1.0 inclusive).
    pub share: f64,

    /// Human-readable name of the payee (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// ── Tolerance ──────────────────────────────────────────────────────────────

/// Tolerance for floating-point comparison of royalty splits.
const SPLIT_TOLERANCE: f64 = 0.001;

// ── Builders ───────────────────────────────────────────────────────────────

impl RightsInfo {
    /// Creates a new empty `RightsInfo`.
    pub fn new() -> Self {
        Self {
            contract_address: None,
            contract_chain: None,
            splits: Vec::new(),
            license_uri: None,
        }
    }

    /// Sets the smart contract address and chain.
    pub fn with_contract(mut self, address: impl Into<String>, chain: impl Into<String>) -> Self {
        self.contract_address = Some(address.into());
        self.contract_chain = Some(chain.into());
        self
    }

    /// Sets the license URI.
    pub fn with_license_uri(mut self, uri: impl Into<String>) -> Self {
        self.license_uri = Some(uri.into());
        self
    }

    /// Adds a royalty split.
    pub fn with_split(mut self, split: RoyaltySplit) -> Self {
        self.splits.push(split);
        self
    }

    /// Adds multiple royalty splits at once.
    pub fn with_splits(mut self, splits: Vec<RoyaltySplit>) -> Self {
        self.splits.extend(splits);
        self
    }

    /// Serializes this rights info to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(MetadataError::from)
    }

    /// Serializes this rights info to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(MetadataError::from)
    }

    /// Deserializes rights info from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(MetadataError::from)
    }

    /// Validates this rights info.
    ///
    /// Checks:
    /// - Each split's share is in [0.0, 1.0]
    /// - Each split's chain is a supported identifier
    /// - Each split's address is not empty
    /// - If splits are present, shares sum to 1.0 (within tolerance)
    /// - If contract_chain is set, it is a supported chain
    pub fn validate(&self) -> Result<()> {
        // Validate contract chain if set
        if let Some(ref chain) = self.contract_chain {
            if !SUPPORTED_CHAINS.contains(&chain.as_str()) {
                return Err(MetadataError::UnsupportedChain(chain.clone()));
            }
        }

        // Validate each split
        for (i, split) in self.splits.iter().enumerate() {
            if split.address.trim().is_empty() {
                return Err(MetadataError::MissingField(format!("splits[{i}].address")));
            }

            if !(0.0..=1.0).contains(&split.share) {
                return Err(MetadataError::InvalidShare(split.share));
            }

            if !SUPPORTED_CHAINS.contains(&split.chain.as_str()) {
                return Err(MetadataError::UnsupportedChain(split.chain.clone()));
            }
        }

        // Validate splits sum to 1.0
        if !self.splits.is_empty() {
            let sum: f64 = self.splits.iter().map(|s| s.share).sum();
            if (sum - 1.0).abs() > SPLIT_TOLERANCE {
                return Err(MetadataError::InvalidSplitSum {
                    sum,
                    tolerance: SPLIT_TOLERANCE,
                });
            }
        }

        Ok(())
    }

    /// Returns the list of supported blockchain identifiers.
    pub fn supported_chains() -> &'static [&'static str] {
        SUPPORTED_CHAINS
    }

    /// Returns the total share allocated across all splits.
    pub fn total_share(&self) -> f64 {
        self.splits.iter().map(|s| s.share).sum()
    }
}

impl Default for RightsInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl RoyaltySplit {
    /// Creates a new royalty split.
    pub fn new(address: impl Into<String>, chain: impl Into<String>, share: f64) -> Self {
        Self {
            address: address.into(),
            chain: chain.into(),
            share,
            name: None,
        }
    }

    /// Sets the human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rights_info_new() {
        let rights = RightsInfo::new();
        assert!(rights.splits.is_empty());
        assert!(rights.contract_address.is_none());
    }

    #[test]
    fn test_rights_builder_chain() {
        let rights = RightsInfo::new()
            .with_contract("0x1234", "ethereum")
            .with_license_uri("https://example.com/license")
            .with_split(RoyaltySplit::new("0xAAAA", "ethereum", 0.6).with_name("Alice"))
            .with_split(RoyaltySplit::new("0xBBBB", "ethereum", 0.4).with_name("Bob"));

        assert_eq!(rights.contract_address.as_deref(), Some("0x1234"));
        assert_eq!(rights.splits.len(), 2);
        assert_eq!(rights.splits[0].name.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_rights_json_round_trip() {
        let rights = RightsInfo::new()
            .with_contract("0x1234", "polygon")
            .with_split(RoyaltySplit::new("0xAAAA", "polygon", 0.5))
            .with_split(RoyaltySplit::new("0xBBBB", "polygon", 0.5));

        let json = rights.to_json_pretty().unwrap();
        let parsed = RightsInfo::from_json(&json).unwrap();
        assert_eq!(rights, parsed);
    }

    #[test]
    fn test_rights_validate_ok() {
        let rights = RightsInfo::new()
            .with_contract("0x1234", "ethereum")
            .with_split(RoyaltySplit::new("0xAAAA", "ethereum", 0.7))
            .with_split(RoyaltySplit::new("0xBBBB", "ethereum", 0.3));

        assert!(rights.validate().is_ok());
    }

    #[test]
    fn test_rights_validate_empty_splits_ok() {
        // Empty splits are valid (no royalties defined yet)
        let rights = RightsInfo::new().with_license_uri("https://example.com");
        assert!(rights.validate().is_ok());
    }

    #[test]
    fn test_rights_validate_bad_sum() {
        let rights = RightsInfo::new()
            .with_split(RoyaltySplit::new("0xAAAA", "ethereum", 0.5))
            .with_split(RoyaltySplit::new("0xBBBB", "ethereum", 0.3));

        let err = rights.validate().unwrap_err();
        assert!(err.to_string().contains("splits sum"));
    }

    #[test]
    fn test_rights_validate_bad_share() {
        let rights = RightsInfo::new().with_split(RoyaltySplit::new("0xAAAA", "ethereum", 1.5));

        let err = rights.validate().unwrap_err();
        assert!(err.to_string().contains("share out of range"));
    }

    #[test]
    fn test_rights_validate_negative_share() {
        let rights = RightsInfo::new().with_split(RoyaltySplit::new("0xAAAA", "ethereum", -0.1));

        assert!(rights.validate().is_err());
    }

    #[test]
    fn test_rights_validate_unsupported_chain() {
        let rights = RightsInfo::new().with_split(RoyaltySplit::new("0xAAAA", "bitcoin", 1.0));

        let err = rights.validate().unwrap_err();
        assert!(err.to_string().contains("unsupported chain"));
    }

    #[test]
    fn test_rights_validate_empty_address() {
        let rights = RightsInfo::new().with_split(RoyaltySplit::new("  ", "ethereum", 1.0));

        let err = rights.validate().unwrap_err();
        assert!(err.to_string().contains("address"));
    }

    #[test]
    fn test_rights_validate_bad_contract_chain() {
        let rights = RightsInfo::new().with_contract("0x1234", "dogecoin");

        let err = rights.validate().unwrap_err();
        assert!(err.to_string().contains("unsupported chain"));
    }

    #[test]
    fn test_rights_total_share() {
        let rights = RightsInfo::new()
            .with_split(RoyaltySplit::new("0xAAAA", "ethereum", 0.6))
            .with_split(RoyaltySplit::new("0xBBBB", "ethereum", 0.4));

        assert!((rights.total_share() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rights_supported_chains() {
        let chains = RightsInfo::supported_chains();
        assert!(chains.contains(&"ethereum"));
        assert!(chains.contains(&"polygon"));
        assert!(chains.contains(&"solana"));
        assert!(!chains.contains(&"bitcoin"));
    }

    #[test]
    fn test_rights_default() {
        let rights = RightsInfo::default();
        assert!(rights.splits.is_empty());
    }
}
