def test_pipeline_config_has_all_llm_max_recovery_attempts():
    from pipeline.config import PipelineConfig
    config = PipelineConfig()
    assert hasattr(config, "all_llm_max_recovery_attempts")
    assert config.all_llm_max_recovery_attempts == 5


def test_pipeline_config_all_llm_field_is_configurable():
    from pipeline.config import PipelineConfig
    config = PipelineConfig(all_llm_max_recovery_attempts=3)
    assert config.all_llm_max_recovery_attempts == 3
