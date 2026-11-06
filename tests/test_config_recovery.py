def test_pipeline_config_has_noskel_max_recovery_attempts():
    from pipeline.config import PipelineConfig
    config = PipelineConfig()
    assert hasattr(config, "noskel_max_recovery_attempts")
    assert config.noskel_max_recovery_attempts == 5


def test_pipeline_config_noskel_field_is_configurable():
    from pipeline.config import PipelineConfig
    config = PipelineConfig(noskel_max_recovery_attempts=3)
    assert config.noskel_max_recovery_attempts == 3
