// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_LIGHT_SCENE_NODE_H_INCLUDED__
#define __C_LIGHT_SCENE_NODE_H_INCLUDED__

#include "ILightSceneNode.h"

namespace irr
{
namespace scene
{

//! Scene node which is a dynamic light. You can switch the light on and off by 
//! making it visible or not, and let it be animated by ordinary scene node animators.
class CLightSceneNode : public ILightSceneNode
{
public:

	//! constructor
	CLightSceneNode(ISceneNode* parent, ISceneManager* mgr, s32 id,
		const core::vector3df& position, video::SColorf color, f32 range);

	virtual ~CLightSceneNode() { }

	//! pre render event
	virtual void OnRegisterSceneNode();

	//! render
	virtual void render();

	//! set node light data from light info
	virtual void setLightData(const video::SLight& light);

	//! \return Returns the light data.
	virtual const video::SLight& getLightData() const;

	//! \return Returns the light data.
	virtual video::SLight& getLightData();

	//! Sets if the node should be visible or not.
	/** All children of this node won't be visible either, when set
	to true.
	\param isVisible If the node shall be visible. */
	virtual void setVisible(bool isVisible);

	//! returns the axis aligned bounding box of this node
	virtual const core::aabbox3d<f32>& getBoundingBox() const;

	//! Returns type of the scene node
	virtual ESCENE_NODE_TYPE getType() const { return ESNT_LIGHT; }

	//! Writes attributes of the scene node.
	virtual void serializeAttributes(io::IAttributes* out, io::SAttributeReadWriteOptions* options=0) const;

	//! Reads attributes of the scene node.
	virtual void deserializeAttributes(io::IAttributes* in, io::SAttributeReadWriteOptions* options=0);

	//! Creates a clone of this scene node and its children.
	virtual ISceneNode* clone(ISceneNode* newParent=0, ISceneManager* newManager=0); 


	//! Sets the light's radius of influence.
	/** Outside this radius the light won't lighten geometry and cast no
	shadows. Setting the radius will also influence the attenuation, setting
	it to (0,1/radius,0). If you want to override this behavior, set the
	attenuation after the radius.
	\param radius The new radius. */
	virtual void setRadius(f32 radius);

	//! Gets the light's radius of influence.
	/** \return The current radius. */
	virtual f32 getRadius() const;

	//! Sets the light type.
	/** \param type The new type. */
	virtual void setLightType(video::E_LIGHT_TYPE type);

	//! Gets the light type.
	/** \return The current light type. */
	virtual video::E_LIGHT_TYPE getLightType() const;

	//! Sets whether this light casts shadows.
	/** Enabling this flag won't automatically cast shadows, the meshes
	will still need shadow scene nodes attached. But one can enable or
	disable distinct lights for shadow casting for performance reasons.
	\param shadow True if this light shall cast shadows. */
	virtual void enableCastShadow(bool shadow=true);

	//! Check whether this light casts shadows.
	/** \return True if light would cast shadows, else false. */
	virtual bool getCastShadow() const;

	//! Refreshes the temporary no-fade override for this light.
	/** The Vulkan renderer fades lights when they enter or leave its active
	light set. Some gameplay-controlled lights, such as nitro and skid
	lights, are switched on and off directly by the game and should change
	immediately instead.

	Call this regularly, typically every update, while the light should
	skip fading. The override expires one second after the last call, so
	normal fading resumes when updates stop. */
	virtual void refreshNoFadeInOut();

	//! Returns whether the temporary no-fade override is active.
	virtual bool isNoFadeInOutActive() const;
private:

	video::SLight LightData;
	core::aabbox3d<f32> BBox;
	s32 DriverLightIndex;
	// Time when the no-fade override was last refreshed, in milliseconds.
	u32 LastFadeSet;
	bool LightIsOn;
	void doLightRecalc();
};


} // end namespace scene
} // end namespace irr

#endif

