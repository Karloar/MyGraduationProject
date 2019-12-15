from django.db import models

# Create your models here.

class SemEval2010Data(models.Model):
    sent = models.CharField(max_length=500)
    entity1_idx = models.IntegerField()
    entity2_idx = models.IntegerField()
    entity1 = models.CharField(max_length=200)
    entity2 = models.CharField(max_length=200)
    relation = models.IntegerField()
    trigger_center = models.IntegerField(null=True)
    is_train = models.BooleanField(default=True)
    trigger_words = models.CharField(max_length=200)


class SemEval2010Relation(models.Model):
    number = models.IntegerField()
    name = models.CharField(max_length=500)
