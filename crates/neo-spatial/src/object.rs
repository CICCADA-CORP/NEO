//! Audio object positioning in 3D space.
//!
//! Implements the spatial scene model from SPECIFICATION.md Section 11.1,
//! with 3D object positioning, keyframe interpolation, room configuration,
//! and JSON serialization for the SPAT chunk.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SpatialError};

/// A 3D position for an audio object using spherical coordinates.
///
/// Follows the coordinate system defined in SPECIFICATION.md Section 11.2:
/// - **Azimuth**: -180° to +180° (0° = front, +90° = right, -90° = left, ±180° = rear)
/// - **Elevation**: -90° to +90° (0° = ear level, +90° = above, -90° = below)
/// - **Distance**: 0.0 to 1.0 (normalized; 0.0 = inside head, 1.0 = maximum distance)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    /// Azimuth angle in degrees (-180 to 180, 0 = front).
    pub azimuth: f64,
    /// Elevation angle in degrees (-90 to 90, 0 = ear level).
    pub elevation: f64,
    /// Distance from listener (0.0 to 1.0, normalized).
    pub distance: f64,
}

impl Position3D {
    /// Creates a new 3D position with validation.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::InvalidAzimuth`] if azimuth is outside -180.0..=180.0.
    /// Returns [`SpatialError::InvalidElevation`] if elevation is outside -90.0..=90.0.
    /// Returns [`SpatialError::InvalidDistance`] if distance is outside 0.0..=1.0.
    pub fn new(azimuth: f64, elevation: f64, distance: f64) -> Result<Self> {
        if !(-180.0..=180.0).contains(&azimuth) {
            return Err(SpatialError::InvalidAzimuth(azimuth));
        }
        if !(-90.0..=90.0).contains(&elevation) {
            return Err(SpatialError::InvalidElevation(elevation));
        }
        if !(0.0..=1.0).contains(&distance) {
            return Err(SpatialError::InvalidDistance(distance));
        }
        Ok(Self {
            azimuth,
            elevation,
            distance,
        })
    }

    /// Creates a position at the front center (0° azimuth, 0° elevation, 0.8 distance).
    pub fn front_center() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 0.8,
        }
    }

    /// Linearly interpolates between this position and another.
    ///
    /// The parameter `t` is clamped to 0.0..=1.0. When `t = 0.0`, returns `self`;
    /// when `t = 1.0`, returns `other`.
    pub fn lerp(&self, other: &Position3D, t: f64) -> Position3D {
        let t = t.clamp(0.0, 1.0);
        Position3D {
            azimuth: self.azimuth + (other.azimuth - self.azimuth) * t,
            elevation: self.elevation + (other.elevation - self.elevation) * t,
            distance: self.distance + (other.distance - self.distance) * t,
        }
    }

    /// Converts the spherical position to Cartesian coordinates (x, y, z).
    ///
    /// - x = forward (positive = front)
    /// - y = right (positive = right)
    /// - z = up (positive = up)
    pub fn to_cartesian(&self) -> (f64, f64, f64) {
        let az_rad = self.azimuth.to_radians();
        let el_rad = self.elevation.to_radians();
        let x = self.distance * az_rad.cos() * el_rad.cos();
        let y = self.distance * az_rad.sin() * el_rad.cos();
        let z = self.distance * el_rad.sin();
        (x, y, z)
    }
}

impl Default for Position3D {
    fn default() -> Self {
        Self::front_center()
    }
}

/// A time-stamped position keyframe for dynamic audio objects.
///
/// Position interpolation between keyframes uses linear interpolation
/// as required by SPECIFICATION.md Section 11.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionKeyframe {
    /// Time in seconds from the start of the audio.
    pub time: f64,
    /// Position at this time.
    pub position: Position3D,
}

impl PositionKeyframe {
    /// Creates a new keyframe at the given time and position.
    pub fn new(time: f64, position: Position3D) -> Self {
        Self { time, position }
    }
}

/// An audio object with spatial positioning metadata.
///
/// Each audio object corresponds to a stem and defines its position
/// in 3D space over time via keyframes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioObject {
    /// Which stem this object corresponds to.
    pub stem_id: u8,
    /// Volume (0.0 to 1.0).
    pub gain: f64,
    /// Spatial extent/spread (0.0 = point source, 1.0 = omnidirectional).
    pub spread: f64,
    /// Position keyframes (interpolated between for smooth movement).
    pub keyframes: Vec<PositionKeyframe>,
}

impl AudioObject {
    /// Creates a new audio object builder for the given stem ID.
    pub fn new(stem_id: u8) -> Self {
        Self {
            stem_id,
            gain: 1.0,
            spread: 0.0,
            keyframes: Vec::new(),
        }
    }

    /// Sets the gain (volume) for this object.
    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Sets the spatial spread for this object.
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }

    /// Adds a position keyframe.
    pub fn with_keyframe(mut self, time: f64, position: Position3D) -> Self {
        self.keyframes.push(PositionKeyframe::new(time, position));
        self
    }

    /// Validates this audio object.
    ///
    /// # Errors
    ///
    /// Returns errors if:
    /// - gain is outside 0.0..=1.0
    /// - spread is outside 0.0..=1.0
    /// - no keyframes are present
    /// - keyframes are not in chronological order
    /// - any keyframe position is invalid
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.gain) {
            return Err(SpatialError::InvalidGain(self.gain));
        }
        if !(0.0..=1.0).contains(&self.spread) {
            return Err(SpatialError::InvalidSpread(self.spread));
        }
        if self.keyframes.is_empty() {
            return Err(SpatialError::NoKeyframes(self.stem_id));
        }
        // Validate chronological ordering
        for i in 1..self.keyframes.len() {
            if self.keyframes[i].time < self.keyframes[i - 1].time {
                return Err(SpatialError::KeyframeOrder(i));
            }
        }
        // Validate all positions
        for kf in &self.keyframes {
            let p = &kf.position;
            Position3D::new(p.azimuth, p.elevation, p.distance)?;
        }
        Ok(())
    }

    /// Interpolates the position at the given time.
    ///
    /// Uses linear interpolation between keyframes as required by the specification.
    /// If `time` is before the first keyframe, returns the first keyframe's position.
    /// If `time` is after the last keyframe, returns the last keyframe's position.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::NoKeyframes`] if there are no keyframes.
    pub fn position_at(&self, time: f64) -> Result<Position3D> {
        if self.keyframes.is_empty() {
            return Err(SpatialError::NoKeyframes(self.stem_id));
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return Ok(self.keyframes[0].position);
        }

        // After last keyframe
        let last = &self.keyframes[self.keyframes.len() - 1];
        if time >= last.time {
            return Ok(last.position);
        }

        // Find the two surrounding keyframes and interpolate
        for i in 1..self.keyframes.len() {
            if time <= self.keyframes[i].time {
                let prev = &self.keyframes[i - 1];
                let next = &self.keyframes[i];
                let duration = next.time - prev.time;
                let t = if duration > 0.0 {
                    (time - prev.time) / duration
                } else {
                    0.0
                };
                return Ok(prev.position.lerp(&next.position, t));
            }
        }

        // Fallback (should not be reached)
        Ok(last.position)
    }
}

/// Room type for spatial audio rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoomType {
    /// Small studio room.
    Studio,
    /// Concert hall.
    Hall,
    /// Church or cathedral.
    Cathedral,
    /// Open outdoor space.
    Outdoor,
    /// No room simulation (anechoic).
    Anechoic,
}

/// Room configuration for spatial audio rendering.
///
/// Specifies the room type and reverb characteristics that affect
/// how spatial audio is perceived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoomConfig {
    /// The type of room.
    #[serde(rename = "type")]
    pub room_type: RoomType,
    /// Reverb time in seconds (T60).
    pub reverb_time: f64,
}

impl RoomConfig {
    /// Creates a new room configuration.
    pub fn new(room_type: RoomType, reverb_time: f64) -> Self {
        Self {
            room_type,
            reverb_time,
        }
    }
}

/// A spatial audio scene containing positioned audio objects.
///
/// This is the top-level container for the SPAT chunk data,
/// matching the JSON structure in SPECIFICATION.md Section 11.1.
///
/// # Example
///
/// ```
/// use neo_spatial::object::{SpatialScene, AudioObject, Position3D, RoomConfig, RoomType};
///
/// let scene = SpatialScene::new()
///     .with_object(
///         AudioObject::new(0)
///             .with_gain(1.0)
///             .with_spread(0.2)
///             .with_keyframe(0.0, Position3D::new(-30.0, 0.0, 0.8).unwrap())
///             .with_keyframe(5.0, Position3D::new(30.0, 10.0, 0.6).unwrap()),
///     )
///     .with_room(RoomConfig::new(RoomType::Studio, 0.3));
///
/// scene.validate().unwrap();
/// let json = scene.to_json().unwrap();
/// let restored = SpatialScene::from_json(&json).unwrap();
/// assert_eq!(scene, restored);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialScene {
    /// The positioned audio objects in this scene.
    pub objects: Vec<AudioObject>,
    /// Optional room configuration for reverb and environment simulation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room: Option<RoomConfig>,
}

impl SpatialScene {
    /// Creates a new empty spatial scene.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            room: None,
        }
    }

    /// Adds an audio object to the scene.
    pub fn with_object(mut self, object: AudioObject) -> Self {
        self.objects.push(object);
        self
    }

    /// Sets the room configuration for the scene.
    pub fn with_room(mut self, room: RoomConfig) -> Self {
        self.room = Some(room);
        self
    }

    /// Validates the entire spatial scene.
    ///
    /// Validates all contained audio objects and their keyframes.
    ///
    /// # Errors
    ///
    /// Returns errors propagated from [`AudioObject::validate`].
    pub fn validate(&self) -> Result<()> {
        for obj in &self.objects {
            obj.validate()?;
        }
        Ok(())
    }

    /// Serializes the scene to a JSON string.
    ///
    /// Produces the JSON format expected by the SPAT chunk.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::SerdeJson`] if serialization fails.
    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }

    /// Deserializes a scene from a JSON string.
    ///
    /// Parses the JSON format used by the SPAT chunk.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::SerdeJson`] if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self> {
        let scene: SpatialScene = serde_json::from_str(json)?;
        Ok(scene)
    }

    /// Interpolates the position for a given stem at a given time.
    ///
    /// Finds the audio object with the matching `stem_id` and interpolates
    /// its position at the given time using linear interpolation between keyframes.
    ///
    /// Returns `None` if no object exists for the given `stem_id`.
    ///
    /// # Errors
    ///
    /// Returns errors propagated from [`AudioObject::position_at`].
    pub fn position_at(&self, stem_id: u8, time: f64) -> Result<Option<Position3D>> {
        for obj in &self.objects {
            if obj.stem_id == stem_id {
                return obj.position_at(time).map(Some);
            }
        }
        Ok(None)
    }
}

impl Default for SpatialScene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position3d_valid() {
        let pos = Position3D::new(0.0, 0.0, 0.5);
        assert!(pos.is_ok());
        let pos = pos.unwrap();
        assert_eq!(pos.azimuth, 0.0);
        assert_eq!(pos.elevation, 0.0);
        assert_eq!(pos.distance, 0.5);
    }

    #[test]
    fn test_position3d_boundary_values() {
        assert!(Position3D::new(-180.0, -90.0, 0.0).is_ok());
        assert!(Position3D::new(180.0, 90.0, 1.0).is_ok());
    }

    #[test]
    fn test_position3d_invalid_azimuth() {
        assert!(matches!(
            Position3D::new(181.0, 0.0, 0.5),
            Err(SpatialError::InvalidAzimuth(_))
        ));
        assert!(matches!(
            Position3D::new(-181.0, 0.0, 0.5),
            Err(SpatialError::InvalidAzimuth(_))
        ));
    }

    #[test]
    fn test_position3d_invalid_elevation() {
        assert!(matches!(
            Position3D::new(0.0, 91.0, 0.5),
            Err(SpatialError::InvalidElevation(_))
        ));
    }

    #[test]
    fn test_position3d_invalid_distance() {
        assert!(matches!(
            Position3D::new(0.0, 0.0, 1.1),
            Err(SpatialError::InvalidDistance(_))
        ));
        assert!(matches!(
            Position3D::new(0.0, 0.0, -0.1),
            Err(SpatialError::InvalidDistance(_))
        ));
    }

    #[test]
    fn test_lerp_endpoints() {
        let a = Position3D::new(-30.0, 0.0, 0.8).unwrap();
        let b = Position3D::new(30.0, 10.0, 0.6).unwrap();

        let at_zero = a.lerp(&b, 0.0);
        assert!((at_zero.azimuth - a.azimuth).abs() < f64::EPSILON);
        assert!((at_zero.elevation - a.elevation).abs() < f64::EPSILON);
        assert!((at_zero.distance - a.distance).abs() < f64::EPSILON);

        let at_one = a.lerp(&b, 1.0);
        assert!((at_one.azimuth - b.azimuth).abs() < f64::EPSILON);
        assert!((at_one.elevation - b.elevation).abs() < f64::EPSILON);
        assert!((at_one.distance - b.distance).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lerp_midpoint() {
        let a = Position3D::new(-30.0, 0.0, 0.8).unwrap();
        let b = Position3D::new(30.0, 10.0, 0.6).unwrap();

        let mid = a.lerp(&b, 0.5);
        assert!((mid.azimuth - 0.0).abs() < f64::EPSILON);
        assert!((mid.elevation - 5.0).abs() < f64::EPSILON);
        assert!((mid.distance - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lerp_clamps_t() {
        let a = Position3D::new(0.0, 0.0, 0.5).unwrap();
        let b = Position3D::new(90.0, 45.0, 1.0).unwrap();

        let clamped_low = a.lerp(&b, -1.0);
        assert!((clamped_low.azimuth - a.azimuth).abs() < f64::EPSILON);

        let clamped_high = a.lerp(&b, 2.0);
        assert!((clamped_high.azimuth - b.azimuth).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_cartesian_front_center() {
        let pos = Position3D::new(0.0, 0.0, 1.0).unwrap();
        let (x, y, z) = pos.to_cartesian();
        assert!((x - 1.0).abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_to_cartesian_right() {
        let pos = Position3D::new(90.0, 0.0, 1.0).unwrap();
        let (x, y, z) = pos.to_cartesian();
        assert!(x.abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_to_cartesian_above() {
        let pos = Position3D::new(0.0, 90.0, 1.0).unwrap();
        let (x, y, z) = pos.to_cartesian();
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!((z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_audio_object_validation_ok() {
        let obj = AudioObject::new(0)
            .with_gain(0.8)
            .with_spread(0.2)
            .with_keyframe(0.0, Position3D::front_center())
            .with_keyframe(5.0, Position3D::new(30.0, 10.0, 0.6).unwrap());
        assert!(obj.validate().is_ok());
    }

    #[test]
    fn test_audio_object_invalid_gain() {
        let obj = AudioObject::new(0)
            .with_gain(1.5)
            .with_keyframe(0.0, Position3D::front_center());
        assert!(matches!(obj.validate(), Err(SpatialError::InvalidGain(_))));
    }

    #[test]
    fn test_audio_object_invalid_spread() {
        let obj = AudioObject::new(0)
            .with_spread(-0.1)
            .with_keyframe(0.0, Position3D::front_center());
        assert!(matches!(
            obj.validate(),
            Err(SpatialError::InvalidSpread(_))
        ));
    }

    #[test]
    fn test_audio_object_no_keyframes() {
        let obj = AudioObject::new(0);
        assert!(matches!(obj.validate(), Err(SpatialError::NoKeyframes(0))));
    }

    #[test]
    fn test_audio_object_keyframe_order() {
        let obj = AudioObject::new(0)
            .with_keyframe(5.0, Position3D::front_center())
            .with_keyframe(2.0, Position3D::front_center()); // out of order
        assert!(matches!(
            obj.validate(),
            Err(SpatialError::KeyframeOrder(1))
        ));
    }

    #[test]
    fn test_position_at_single_keyframe() {
        let obj = AudioObject::new(0).with_keyframe(1.0, Position3D::new(30.0, 0.0, 0.5).unwrap());
        let pos = obj.position_at(0.0).unwrap();
        assert!((pos.azimuth - 30.0).abs() < f64::EPSILON);
        let pos = obj.position_at(5.0).unwrap();
        assert!((pos.azimuth - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position_at_interpolation() {
        let obj = AudioObject::new(0)
            .with_keyframe(0.0, Position3D::new(-30.0, 0.0, 0.8).unwrap())
            .with_keyframe(10.0, Position3D::new(30.0, 10.0, 0.6).unwrap());

        let pos = obj.position_at(5.0).unwrap();
        assert!((pos.azimuth - 0.0).abs() < 1e-10);
        assert!((pos.elevation - 5.0).abs() < 1e-10);
        assert!((pos.distance - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_position_at_before_first() {
        let obj = AudioObject::new(0).with_keyframe(2.0, Position3D::new(45.0, 0.0, 1.0).unwrap());
        let pos = obj.position_at(0.0).unwrap();
        assert!((pos.azimuth - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position_at_after_last() {
        let obj = AudioObject::new(0)
            .with_keyframe(0.0, Position3D::new(-45.0, 0.0, 0.5).unwrap())
            .with_keyframe(2.0, Position3D::new(45.0, 0.0, 1.0).unwrap());
        let pos = obj.position_at(10.0).unwrap();
        assert!((pos.azimuth - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spatial_scene_round_trip_json() {
        let scene = SpatialScene::new()
            .with_object(
                AudioObject::new(0)
                    .with_gain(1.0)
                    .with_spread(0.2)
                    .with_keyframe(0.0, Position3D::new(-30.0, 0.0, 0.8).unwrap())
                    .with_keyframe(5.0, Position3D::new(30.0, 10.0, 0.6).unwrap()),
            )
            .with_room(RoomConfig::new(RoomType::Studio, 0.3));

        let json = scene.to_json().unwrap();
        let restored = SpatialScene::from_json(&json).unwrap();
        assert_eq!(scene, restored);
    }

    #[test]
    fn test_spatial_scene_json_matches_spec() {
        // Verify the JSON structure matches SPECIFICATION.md Section 11.1
        let scene = SpatialScene::new()
            .with_object(
                AudioObject::new(0)
                    .with_gain(1.0)
                    .with_spread(0.2)
                    .with_keyframe(0.0, Position3D::new(-30.0, 0.0, 0.8).unwrap())
                    .with_keyframe(5.0, Position3D::new(30.0, 10.0, 0.6).unwrap()),
            )
            .with_room(RoomConfig::new(RoomType::Studio, 0.3));

        let json = scene.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Check top-level structure
        assert!(parsed["objects"].is_array());
        assert!(parsed["room"].is_object());

        // Check object fields
        let obj = &parsed["objects"][0];
        assert_eq!(obj["stem_id"], 0);
        assert_eq!(obj["gain"], 1.0);
        assert_eq!(obj["spread"], 0.2);

        // Check keyframe fields
        let kf0 = &obj["keyframes"][0];
        assert_eq!(kf0["time"], 0.0);
        assert_eq!(kf0["position"]["azimuth"], -30.0);

        // Check room fields
        assert_eq!(parsed["room"]["type"], "studio");
        assert_eq!(parsed["room"]["reverb_time"], 0.3);
    }

    #[test]
    fn test_spatial_scene_no_room() {
        let scene = SpatialScene::new()
            .with_object(AudioObject::new(0).with_keyframe(0.0, Position3D::front_center()));
        let json = scene.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        // room should not be present
        assert!(parsed.get("room").is_none());
    }

    #[test]
    fn test_spatial_scene_position_at() {
        let scene = SpatialScene::new()
            .with_object(
                AudioObject::new(0).with_keyframe(0.0, Position3D::new(-30.0, 0.0, 0.8).unwrap()),
            )
            .with_object(
                AudioObject::new(1).with_keyframe(0.0, Position3D::new(30.0, 0.0, 0.8).unwrap()),
            );

        let pos0 = scene.position_at(0, 0.0).unwrap();
        assert!(pos0.is_some());
        assert!((pos0.unwrap().azimuth - (-30.0)).abs() < f64::EPSILON);

        let pos1 = scene.position_at(1, 0.0).unwrap();
        assert!(pos1.is_some());
        assert!((pos1.unwrap().azimuth - 30.0).abs() < f64::EPSILON);

        // Non-existent stem
        let pos2 = scene.position_at(2, 0.0).unwrap();
        assert!(pos2.is_none());
    }

    #[test]
    fn test_spatial_scene_validate() {
        let scene = SpatialScene::new().with_object(
            AudioObject::new(0)
                .with_gain(1.0)
                .with_spread(0.2)
                .with_keyframe(0.0, Position3D::front_center()),
        );
        assert!(scene.validate().is_ok());
    }

    #[test]
    fn test_spatial_scene_validate_fails_on_invalid_object() {
        let scene = SpatialScene::new().with_object(
            AudioObject::new(0).with_gain(2.0), // invalid gain, no keyframes
        );
        assert!(scene.validate().is_err());
    }

    #[test]
    fn test_position3d_default() {
        let pos = Position3D::default();
        assert!((pos.azimuth - 0.0).abs() < f64::EPSILON);
        assert!((pos.elevation - 0.0).abs() < f64::EPSILON);
        assert!((pos.distance - 0.8).abs() < f64::EPSILON);
    }
}
